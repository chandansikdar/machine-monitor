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
            df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True, format="mixed")
            df = df.rename(columns={ts_col: "timestamp"}).sort_values("timestamp")

            # Persist to disk
            machine_dir = self.data_dir / machine_id
            machine_dir.mkdir(parents=True, exist_ok=True)
            stem = filename.rsplit(".", 1)[0]
            file_path = machine_dir / f"{stem}.csv"
            df.to_csv(file_path, index=False)

            # Remove any existing DB rows for this filename before inserting
            self.conn.execute(
                "DELETE FROM data_files WHERE machine_id = ? AND file_path LIKE ?",
                [machine_id, f"%{stem}.csv"],
            )

            # Register in metadata table
            col_list = ",".join(df.columns.tolist())
            self.conn.execute(
                "INSERT INTO data_files (machine_id, file_path, rows, columns) VALUES (?, ?, ?, ?)",
                [machine_id, str(file_path), len(df), col_list],
            )

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
                FROM read_csv_auto('{pattern}', union_by_name=true)
                ORDER BY timestamp
                {f"LIMIT {limit}" if limit > 0 else ""}
            """).df()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            # Coerce any numeric-looking columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            return None

    def get_data_from_file(self, machine_id: str, filename: str) -> Optional[pd.DataFrame]:
        """Load data from a specific file only (not all files for the machine)."""
        machine_dir = self.data_dir / machine_id
        # Find the matching file
        matches = list(machine_dir.glob("*.csv")) if machine_dir.exists() else []
        target = next((f for f in matches if f.name == filename or filename in f.name), None)
        if not target or not target.exists():
            return None
        try:
            df = self.conn.execute(f"""
                SELECT * FROM read_csv_auto('{str(target).replace(chr(92),"/")}')
                ORDER BY timestamp
            """).df()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            return None

    def get_file_info(self, machine_id: str) -> list:
        rows = self.conn.execute("""
            SELECT file_path, rows, columns, ingested_at
            FROM data_files WHERE machine_id = ?
            ORDER BY ingested_at
        """, [machine_id]).fetchall()
        # Deduplicate by filename — keep latest entry for each name
        seen = {}
        for r in rows:
            name = Path(r[0]).name
            seen[name] = {"file": name, "file_path": r[0], "rows": r[1], "columns": r[2], "ingested_at": r[3]}
        return list(seen.values())

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

    def delete_log(self, machine_id: str, filename: str):
        self.conn.execute(
            "DELETE FROM maintenance_logs WHERE machine_id = ? AND filename = ?",
            [machine_id, filename],
        )
