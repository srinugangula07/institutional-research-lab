import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DB_PATH = os.getenv(
    "JOURNAL_DB_PATH",
    "/tmp/institutional_market_journal.db",
)


def get_db_path():
    return Path(DEFAULT_DB_PATH)


def _json_safe(v):
    if pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, np.generic):
        return v.item()
    return v


def init_journal_db(db_path=None):
    path = Path(db_path or get_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_time TEXT NOT NULL,
                snapshot_type TEXT NOT NULL,
                stock TEXT,
                payload_json TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_snapshots_type_time
            ON snapshots(snapshot_type, snapshot_time)
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_snapshots_stock_time
            ON snapshots(stock, snapshot_time)
            """
        )

    return str(path)


def append_snapshot(snapshot_type, df, db_path=None, snapshot_time=None):
    """
    Append one dataframe snapshot to SQLite.

    Each row is stored as JSON payload so the schema can evolve without
    destructive DB migrations while the trading engine is still under development.
    """
    if df is None or df.empty:
        return 0

    path = init_journal_db(db_path)
    ts = pd.Timestamp.now() if snapshot_time is None else pd.Timestamp(snapshot_time)

    work = df.copy()

    rows = []
    for _, r in work.iterrows():
        payload = {
            str(k): _json_safe(v)
            for k, v in r.to_dict().items()
        }
        stock = payload.get("Stock")
        rows.append(
            (
                ts.isoformat(),
                str(snapshot_type),
                None if stock is None else str(stock),
                json.dumps(payload, ensure_ascii=False),
            )
        )

    with sqlite3.connect(path) as con:
        con.executemany(
            """
            INSERT INTO snapshots(
                snapshot_time,
                snapshot_type,
                stock,
                payload_json
            )
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )

    return len(rows)


def snapshot_counts(db_path=None):
    path = init_journal_db(db_path)

    with sqlite3.connect(path) as con:
        q = pd.read_sql_query(
            """
            SELECT
                snapshot_type,
                COUNT(*) AS rows,
                COUNT(DISTINCT stock) AS stocks,
                MIN(snapshot_time) AS first_time,
                MAX(snapshot_time) AS last_time
            FROM snapshots
            GROUP BY snapshot_type
            ORDER BY snapshot_type
            """,
            con,
        )

    return q


def read_snapshots(snapshot_type=None, stock=None, limit=5000, db_path=None):
    path = init_journal_db(db_path)

    where = []
    params = []

    if snapshot_type:
        where.append("snapshot_type = ?")
        params.append(str(snapshot_type))

    if stock:
        where.append("stock = ?")
        params.append(str(stock).upper())

    sql = """
        SELECT id, snapshot_time, snapshot_type, stock, payload_json
        FROM snapshots
    """

    if where:
        sql += " WHERE " + " AND ".join(where)

    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    with sqlite3.connect(path) as con:
        raw = pd.read_sql_query(sql, con, params=params)

    if raw.empty:
        return raw

    payloads = raw["payload_json"].apply(json.loads)
    payload_df = pd.json_normalize(payloads)

    out = pd.concat(
        [
            raw[["id", "snapshot_time", "snapshot_type", "stock"]].reset_index(drop=True),
            payload_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Avoid duplicate Stock field if JSON payload already contains it.
    if "Stock" in out.columns and "stock" in out.columns:
        out["Stock"] = out["Stock"].fillna(out["stock"])

    return out


def database_size_bytes(db_path=None):
    path = Path(db_path or get_db_path())
    return path.stat().st_size if path.exists() else 0


def clear_journal_db(db_path=None):
    path = init_journal_db(db_path)
    with sqlite3.connect(path) as con:
        con.execute("DELETE FROM snapshots")
        con.execute("VACUUM")


def export_db_bytes(db_path=None):
    path = Path(init_journal_db(db_path))
    return path.read_bytes()


def restore_db_bytes(data, db_path=None):
    path = Path(db_path or get_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".restore_tmp")
    tmp.write_bytes(data)

    # Validate the uploaded file is a readable SQLite database.
    with sqlite3.connect(tmp) as con:
        con.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()

    tmp.replace(path)
    init_journal_db(path)
    return str(path)


def init_capture_registry(db_path=None):
    path = Path(db_path or get_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS capture_registry (
                capture_key TEXT PRIMARY KEY,
                capture_time TEXT NOT NULL,
                snapshot_type TEXT NOT NULL,
                rows_saved INTEGER NOT NULL
            )
            """
        )

    return str(path)


def capture_exists(capture_key, db_path=None):
    path = init_capture_registry(db_path)

    with sqlite3.connect(path) as con:
        row = con.execute(
            """
            SELECT 1
            FROM capture_registry
            WHERE capture_key = ?
            LIMIT 1
            """,
            (str(capture_key),),
        ).fetchone()

    return row is not None


def append_snapshot_once(
    snapshot_type,
    df,
    capture_key,
    db_path=None,
    snapshot_time=None,
):
    """
    Append a snapshot only once for a unique capture_key.

    Returns:
      rows_saved, status
      status in {"SAVED", "ALREADY_CAPTURED", "EMPTY"}
    """
    if df is None or df.empty:
        return 0, "EMPTY"

    path = init_journal_db(db_path)
    init_capture_registry(path)

    key = str(capture_key)

    if capture_exists(key, path):
        return 0, "ALREADY_CAPTURED"

    ts = pd.Timestamp.now() if snapshot_time is None else pd.Timestamp(snapshot_time)

    # First save rows, then register capture key.
    rows_saved = append_snapshot(
        snapshot_type=snapshot_type,
        df=df,
        db_path=path,
        snapshot_time=ts,
    )

    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT OR IGNORE INTO capture_registry(
                capture_key,
                capture_time,
                snapshot_type,
                rows_saved
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                key,
                ts.isoformat(),
                str(snapshot_type),
                int(rows_saved),
            ),
        )

    return int(rows_saved), "SAVED"


def capture_registry(db_path=None):
    path = init_capture_registry(db_path)

    with sqlite3.connect(path) as con:
        q = pd.read_sql_query(
            """
            SELECT
                capture_key,
                capture_time,
                snapshot_type,
                rows_saved
            FROM capture_registry
            ORDER BY capture_time DESC
            """,
            con,
        )

    return q
