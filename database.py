"""
database.py — Data persistence layer (DuckDB + flat CSV files)

Architecture:
  - machines table: machine profiles and specs
  - data_files table: registry of every uploaded CSV/Excel file
  - analysis_history table: past Claude analysis results (JSON)
  - Raw sensor data lives as CSV files under data/<machine_id>/*.csv
    DuckDB reads them natively with read_csv_auto(), unioning files as needed.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class Database:
    def __init__(self, db_path: str = "machine_analytics_v2.duckdb", data_dir: str = "data_v2"):
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                machine_id   VARCHAR PRIMARY KEY,
                machine_type VARCHAR NOT NULL,
                description  TEXT,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_files (
                machine_id  VARCHAR,
                file_path   VARCHAR,
                rows        INTEGER,
                columns     VARCHAR,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                machine_id    VARCHAR,
                analysis_type VARCHAR,
                insights      JSON,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                machine_id   VARCHAR,
                filename     VARCHAR,
                file_type    VARCHAR,
                content      TEXT,
                uploaded_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS electrical_baselines (
                machine_id    VARCHAR PRIMARY KEY,
                baseline_json TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    # ------------------------------------------------------------------ #
    # Machine management
    # ------------------------------------------------------------------ #

    def register_machine(self, machine_id: str, machine_type: str, description: str = ""):
        self.conn.execute("""
            INSERT INTO machines (machine_id, machine_type, description)
            VALUES (?, ?, ?)
            ON CONFLICT (machine_id) DO UPDATE SET
                machine_type  = excluded.machine_type,
                description   = excluded.description
        """, [machine_id, machine_type, description])

    def get_machines(self) -> list:
        rows = self.conn.execute(
            "SELECT machine_id, machine_type FROM machines ORDER BY registered_at"
        ).fetchall()
        return [{"machine_id": r[0], "machine_type": r[1]} for r in rows]

    def get_machine_info(self, machine_id: str) -> dict:
        r = self.conn.execute(
            "SELECT machine_id, machine_type, description FROM machines WHERE machine_id = ?",
            [machine_id],
        ).fetchone()
        return (
            {"machine_id": r[0], "machine_type": r[1], "description": r[2] or ""}
            if r else {}
        )

    # ------------------------------------------------------------------ #
    # Data ingestion
    # ------------------------------------------------------------------ #

    def ingest_file(self, file, machine_id: str) -> dict:
        """
        Read an uploaded CSV or Excel file, normalise the timestamp column,
        save it to disk, and register it in data_files.
        """
        try:
            filename = file.name
            if filename.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            if df.empty:
                return {"success": False, "error": "File is empty."}

            # Auto-detect the timestamp column by name
            ts_col = next(
                (c for c in df.columns
                 if any(kw in c.lower() for kw in ["time", "date", "timestamp", "ts"])),
                df.columns[0],
            )

            # Detect timestamp format from raw string values and parse accordingly.
            #
            # Priority 1 — ISO 8601 (YYYY-MM-DD …):
            #   If the first non-null value starts with a 4-digit year followed by
            #   '-', the format is unambiguously ISO.  Use an explicit strptime format
            #   so pandas never attempts day/month inference.  dayfirst is irrelevant
            #   in this path.
            #
            # Priority 2 — Ambiguous DD/MM or MM/DD:
            #   Scan for an unambiguous row where the leading token > 12 (must be a
            #   day → dayfirst=True) or the second token > 12 (must be a day →
            #   dayfirst=False, i.e. month-first).  Default to dayfirst=True
            #   (European standard) only when every row is ambiguous.
            #
            # Note: the former regex r"(\d{1,2})[\/\-\.\s](\d{1,2})" failed silently
            # on ISO strings because the leading token is 4 digits, causing the regex
            # to never match and leaving dayfirst=True, which then flipped April dates
            # (04-01 → 01-04, etc.) when day ≤ 12.
            import re as _re
            _sample_vals = [
                str(v).strip() for v in df[ts_col].dropna().astype(str)
                if str(v).strip()
            ]
            _iso_re   = _re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
            _ambig_re = _re.compile(r"^(\d{1,2})[\/\-\.\s](\d{1,2})")
            _fmt      = None    # explicit strptime format if ISO detected
            _dayfirst = True    # fallback for ambiguous DD/MM

            for _val in _sample_vals:
                if _iso_re.match(_val):
                    # ISO format: month is always the middle token — dayfirst=False
                    _dayfirst = False
                    # Detect whether a time component is present for the format string
                    _fmt = "%Y-%m-%d %H:%M:%S" if " " in _val else "%Y-%m-%d"
                    break
                _m2 = _ambig_re.match(_val)
                if _m2:
                    _first, _second = int(_m2.group(1)), int(_m2.group(2))
                    if _first > 12:
                        _dayfirst = True
                        break
                    if _second > 12:
                        _dayfirst = False
                        break

            if _fmt:
                # Explicit ISO format — no ambiguity possible
                df[ts_col] = pd.to_datetime(df[ts_col], format=_fmt, errors="coerce")
            else:
                df[ts_col] = pd.to_datetime(
                    df[ts_col], dayfirst=_dayfirst, format="mixed", errors="coerce"
                )
            df = df.rename(columns={ts_col: "timestamp"})
            # Drop empty rows — null timestamp means a blank/trailing row in the CSV
            df = df[df["timestamp"].notna()].copy()
            df = df.sort_values("timestamp")

            # Persist to disk
            machine_dir = self.data_dir / machine_id
            machine_dir.mkdir(parents=True, exist_ok=True)
            stem = filename.rsplit(".", 1)[0]
            file_path = machine_dir / f"{stem}.csv"
            df.to_csv(file_path, index=False)

            # Remove ALL existing DB rows for this exact file_path before inserting
            self.conn.execute(
                "DELETE FROM data_files WHERE machine_id = ? AND file_path = ?",
                [machine_id, str(file_path)],
            )

            # Register in metadata table
            col_list = ",".join(df.columns.tolist())
            self.conn.execute(
                "INSERT INTO data_files (machine_id, file_path, rows, columns) VALUES (?, ?, ?, ?)",
                [machine_id, str(file_path), len(df), col_list],
            )
            self.conn.commit()

            return {
                "success": True,
                "rows": len(df),
                "columns": df.columns.tolist(),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Data retrieval
    # ------------------------------------------------------------------ #

    def get_data(self, machine_id: str, limit: int = 0) -> Optional[pd.DataFrame]:
        """
        Return all uploaded data for a machine as a DataFrame indexed by timestamp.
        DuckDB unions multiple CSV files automatically.
        """
        machine_dir = self.data_dir / machine_id
        if not machine_dir.exists() or not list(machine_dir.glob("*.csv")):
            return None
        try:
            pattern = str(machine_dir / "*.csv").replace("\\", "/")
            df = self.conn.execute(f"""
                SELECT *
                FROM read_csv_auto('{pattern}', union_by_name=true, all_varchar=true)
                ORDER BY timestamp
                {f"LIMIT {limit}" if limit > 0 else ""}
            """).df()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df = self._fix_swapped_month_day(df)
            df = df.sort_index()          # re-sort after date correction
            df = self._coerce_numeric(df)
            return df
        except Exception:
            return None

    def get_data_from_file(self, machine_id: str, filename: str) -> Optional[pd.DataFrame]:
        """Load data from a specific file only (not all files for the machine)."""
        machine_dir = self.data_dir / machine_id
        matches = list(machine_dir.glob("*.csv")) if machine_dir.exists() else []
        target = next((f for f in matches if f.name == filename or filename in f.name), None)
        if not target or not target.exists():
            return None
        try:
            df = self.conn.execute(f"""
                SELECT * FROM read_csv_auto('{str(target).replace(chr(92),"/")}',
                all_varchar=true)
                ORDER BY timestamp
            """).df()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            fixed = self._fix_swapped_month_day(df)
            was_fixed = not fixed.index.equals(df.index)
            fixed = fixed.sort_index()     # re-sort after date correction
            fixed = self._coerce_numeric(fixed)
            # Re-save the corrected CSV so the on-disk file is also fixed permanently
            if was_fixed:
                try:
                    corrected = fixed.reset_index()
                    # Drop sidecar columns before saving
                    corrected = corrected[[c for c in corrected.columns
                                          if not c.startswith("_")]]
                    corrected.to_csv(target, index=False)
                except Exception:
                    pass   # Non-critical — data is correct in memory even if save fails
            return fixed
        except Exception:
            return None

    def _fix_swapped_month_day(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Detect and correct timestamps stored with MM-DD instead of DD-MM.

        When pandas parses DD/MM/YYYY with format='mixed' and day <= 12, it may
        silently parse as MM/DD, storing e.g. 01/04/2026 as 2026-01-04 instead
        of 2026-04-01.

        Detection: if >70% of rows share the same day-of-month value AND that
        value is <= 12, it is very likely month and day were swapped.

        Fix: swap month and day for all rows. Rows where the swap is invalid
        (e.g. original day=31 which cannot be a month) are kept as-is.
        """
        import pandas as _pd
        if not isinstance(df.index, _pd.DatetimeIndex):
            return df
        ts = _pd.Series(df.index)
        day_vals = ts.dt.day
        if len(day_vals) == 0:
            return df
        most_common_day = int(day_vals.mode()[0])
        frac = float((day_vals == most_common_day).mean())
        if not (frac > 0.70 and most_common_day <= 12):
            return df   # Dates look correct — no fix needed
        def _swap(t):
            try:
                return t.replace(month=t.day, day=t.month)
            except ValueError:
                return t   # Swap would produce invalid date — keep original
        new_ts = ts.map(_swap)
        df = df.copy()
        df.index = _pd.DatetimeIndex(new_ts.values, name="timestamp")
        return df

    def _coerce_numeric(self, df) -> "pd.DataFrame":
        """Coerce measurement columns to numeric, storing non-numeric cell info.

        Non-numeric cells (e.g. '12:00 AM', 'N/A', 'error') are coerced to NaN.
        Two sidecar columns are added per affected measurement column:
          _non_numeric_flags       : pipe-separated "col:value" strings per row
          _orig_<col>              : original string value for rows with bad data
        This allows the failure report to show the original bad value, not NaN.
        """
        import pandas as _pd
        MEAS = [
            "phase_1_voltage", "phase_2_voltage", "phase_3_voltage",
            "phase_1_current", "phase_2_current", "phase_3_current",
            "phase_1_active_power", "phase_2_active_power", "phase_3_active_power",
        ]
        flags = _pd.Series("", index=df.index)
        for col in df.columns:
            if col in MEAS:
                orig    = df[col]
                coerced = _pd.to_numeric(orig, errors="coerce")
                non_num = coerced.isna() & ~orig.isna()
                if non_num.any():
                    bad_vals = orig[non_num].astype(str)
                    # Store original bad values in a per-column sidecar
                    df[f"_orig_{col}"] = _pd.Series("", index=df.index)
                    for idx in bad_vals.index:
                        entry = f"{col}:{bad_vals[idx]}"
                        flags[idx] = (flags[idx] + "|" + entry).lstrip("|")
                        df.at[idx, f"_orig_{col}"] = bad_vals[idx]
                df[col] = coerced
            else:
                # Non-measurement columns — coerce quietly
                try:
                    df[col] = _pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass
        if flags.any():
            df["_non_numeric_flags"] = flags
        return df

    def get_file_info(self, machine_id: str) -> list:
        rows = self.conn.execute("""
            SELECT file_path, rows, columns, ingested_at
            FROM data_files WHERE machine_id = ?
            ORDER BY ingested_at
        """, [machine_id]).fetchall()

        # Deduplicate by filename — keep latest row per name
        seen = {}
        for r in rows:
            name = Path(r[0]).name
            seen[name] = (r[0], r[1], r[2], r[3])

        # If duplicates existed, rebuild the table rows cleanly
        if len(rows) > len(seen):
            self.conn.execute(
                "DELETE FROM data_files WHERE machine_id = ?", [machine_id]
            )
            for name, (fp, nrows, cols, ingested_at) in seen.items():
                self.conn.execute(
                    "INSERT INTO data_files (machine_id, file_path, rows, columns, ingested_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [machine_id, fp, nrows, cols, ingested_at]
                )
            self.conn.commit()

        return [
            {"file": name, "file_path": fp, "rows": nrows, "columns": cols, "ingested_at": ingested_at}
            for name, (fp, nrows, cols, ingested_at) in seen.items()
        ]

    # ------------------------------------------------------------------ #
    # Analysis history
    # ------------------------------------------------------------------ #

    def save_analysis(self, machine_id: str, analysis_type: str, insights: dict):
        self.conn.execute(
            "INSERT INTO analysis_history (machine_id, analysis_type, insights) VALUES (?, ?, ?)",
            [machine_id, analysis_type, json.dumps(insights)],
        )

    def get_analysis_history(self, machine_id: str) -> list:
        rows = self.conn.execute("""
            SELECT analysis_type, insights, created_at
            FROM analysis_history
            WHERE machine_id = ?
            ORDER BY created_at DESC
            LIMIT 30
        """, [machine_id]).fetchall()
        return [
            {"analysis_type": r[0], "insights": json.loads(r[1]), "timestamp": r[2]}
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Maintenance logs
    # ------------------------------------------------------------------ #

    def save_log(self, machine_id: str, filename: str, file_type: str, content: str):
        self.conn.execute(
            "INSERT INTO maintenance_logs (machine_id, filename, file_type, content) VALUES (?, ?, ?, ?)",
            [machine_id, filename, file_type, content],
        )

    def get_logs(self, machine_id: str) -> list:
        rows = self.conn.execute("""
            SELECT filename, file_type, content, uploaded_at
            FROM maintenance_logs
            WHERE machine_id = ?
            ORDER BY uploaded_at DESC
            LIMIT 20
        """, [machine_id]).fetchall()
        return [
            {"filename": r[0], "file_type": r[1], "content": r[2], "uploaded_at": r[3]}
            for r in rows
        ]

    def get_logs_text(self, machine_id: str) -> str:
        """Return all log content concatenated — for inclusion in Claude prompts."""
        logs = self.get_logs(machine_id)
        if not logs:
            return ""
        parts = []
        for log in logs:
            parts.append(f"--- {log['filename']} (uploaded {log['uploaded_at']}) ---\n{log['content']}")
        return "\n\n".join(parts)

    def delete_file(self, machine_id: str, filename: str):
        """Delete a specific ingested data file and remove its rows from data."""
        try:
            # Find the file path
            rows = self.conn.execute(
                "SELECT file_path FROM data_files WHERE machine_id=? AND file_path LIKE ?",
                [machine_id, f"%{filename}%"]
            ).fetchall()
            for row in rows:
                fp = Path(row[0])
                if fp.exists():
                    fp.unlink()
            self.conn.execute(
                "DELETE FROM data_files WHERE machine_id=? AND file_path LIKE ?",
                [machine_id, f"%{filename}%"]
            )
            return True
        except Exception as e:
            return False

    def delete_all_files(self, machine_id: str):
        """Delete all ingested data files for a machine."""
        try:
            rows = self.conn.execute(
                "SELECT file_path FROM data_files WHERE machine_id=?",
                [machine_id]
            ).fetchall()
            for row in rows:
                fp = Path(row[0])
                if fp.exists():
                    fp.unlink()
            self.conn.execute(
                "DELETE FROM data_files WHERE machine_id=?",
                [machine_id]
            )
            return True
        except Exception as e:
            return False

    def delete_machine(self, machine_id: str):
        """Delete a machine and all its data, logs, and analysis history."""
        try:
            # Delete all data files from disk
            self.delete_all_files(machine_id)
            # Delete from all tables
            for table in ["machines", "data_files", "analysis_history", "maintenance_logs"]:
                self.conn.execute(f"DELETE FROM {table} WHERE machine_id = ?", [machine_id])
            return True
        except Exception as e:
            return False

    # ------------------------------------------------------------------ #
    # Electrical baseline
    # ------------------------------------------------------------------ #

    def save_baseline(self, machine_id: str, baseline_dict: dict):
        """Persist a serialised BaselineMetadata dict for a machine.
        Overwrites any previously stored baseline for this machine.
        """
        self.conn.execute(
            "DELETE FROM electrical_baselines WHERE machine_id = ?", [machine_id]
        )
        self.conn.execute(
            "INSERT INTO electrical_baselines (machine_id, baseline_json) VALUES (?, ?)",
            [machine_id, json.dumps(baseline_dict, default=str)],
        )
        self.conn.commit()

    def get_baseline(self, machine_id: str) -> dict | None:
        """Return the stored baseline dict, or None if none exists."""
        row = self.conn.execute(
            "SELECT baseline_json, created_at FROM electrical_baselines WHERE machine_id = ?",
            [machine_id],
        ).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        d["_stored_at"] = str(row[1])
        return d

    def delete_baseline(self, machine_id: str):
        """Delete the stored baseline for a machine."""
        self.conn.execute(
            "DELETE FROM electrical_baselines WHERE machine_id = ?", [machine_id]
        )
        self.conn.commit()

    def delete_log(self, machine_id: str, filename: str):
        self.conn.execute(
            "DELETE FROM maintenance_logs WHERE machine_id = ? AND filename = ?",
            [machine_id, filename],
        )
