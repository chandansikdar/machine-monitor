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
    def __init__(self, db_path: str = "machine_analytics.duckdb", data_dir: str = "data"):
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
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.rename(columns={ts_col: "timestamp"}).sort_values("timestamp")

            # Persist to disk
            machine_dir = self.data_dir / machine_id
            machine_dir.mkdir(parents=True, exist_ok=True)
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = filename.rsplit(".", 1)[0]
            file_path = machine_dir / f"{tag}_{stem}.csv"
            df.to_csv(file_path, index=False)

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

    def get_data(self, machine_id: str, limit: int = 10_000) -> Optional[pd.DataFrame]:
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
                LIMIT {limit}
            """).df()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            # Coerce any numeric-looking columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            return df
        except Exception:
            return None

    def get_file_info(self, machine_id: str) -> list:
        rows = self.conn.execute("""
            SELECT file_path, rows, columns, ingested_at
            FROM data_files WHERE machine_id = ?
            ORDER BY ingested_at
        """, [machine_id]).fetchall()
        return [
            {"file": Path(r[0]).name, "rows": r[1], "columns": r[2], "ingested_at": r[3]}
            for r in rows
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
