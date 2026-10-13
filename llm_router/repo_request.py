"""SQLite repository: all SQL commands for the requests table."""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, fields
from typing import Any, Optional

from .models import RequestRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL    NOT NULL,
    request_id        TEXT,
    endpoint          TEXT,
    model             TEXT,
    machine           TEXT,
    priority          TEXT,
    stream            INTEGER,
    status            INTEGER,
    queue_wait_ms     REAL,
    ttft_ms           REAL,
    total_ms          REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    prefill_tps       REAL,
    gen_tps           REAL,
    metrics_source    TEXT,
    error             TEXT,
    client                  TEXT,
    client_ip               TEXT,
    request_time            REAL,
    total_duration_request  REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_mm ON requests(machine, model);
"""

_RECORD_FIELDS = {f.name for f in fields(RequestRecord)}


class RequestRepository:
    """Handles all SQLite persistence and queries for request records."""

    def __init__(self, db_path: str) -> None:
        """Opens or creates the SQLite database and applies the schema."""
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Creates tables and indexes; migrates legacy databases that predate the client columns."""
        self._db.executescript(SCHEMA)
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(requests)")}
        migrations: dict[str, str] = {
            "client": "TEXT",
            "client_ip": "TEXT",
            "request_time": "REAL",
            "total_duration_request": "REAL",
        }
        for col, col_type in migrations.items():
            if col not in cols:
                self._db.execute(f"ALTER TABLE requests ADD COLUMN {col} {col_type}")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status)")
        self._db.commit()

    def _row_to_record(self, row: dict[str, Any]) -> RequestRecord:
        """Converts a raw DB row (which includes the auto-increment id) to a RequestRecord."""
        return RequestRecord(**{k: v for k, v in row.items() if k in _RECORD_FIELDS})

    def insert(self, rec: RequestRecord) -> None:
        """Inserts a request record into the database."""
        row = asdict(rec)
        with self._lock:
            cols = ",".join(row)
            self._db.execute(
                f"INSERT INTO requests ({cols}) VALUES ({','.join('?' * len(row))})",
                [int(v) if isinstance(v, bool) else v for v in row.values()],
            )
            self._db.commit()

    def stats(
        self, since: float, model: Optional[str], machine: Optional[str]
    ) -> list[dict[str, Any]]:
        """Returns aggregated per-machine/model performance stats for a time window."""
        where: list[str] = ["ts >= ?"]
        args: list[Any] = [since]
        if model:
            where.append("model = ?")
            args.append(model)
        if machine:
            where.append("machine = ?")
            args.append(machine)
        sql = f"""
            SELECT machine, model, COUNT(*) AS requests,
                   SUM(status >= 400) AS errors,
                   ROUND(AVG(queue_wait_ms), 1)     AS avg_queue_wait_ms,
                   ROUND(MAX(queue_wait_ms), 1)     AS max_queue_wait_ms,
                   ROUND(AVG(ttft_ms), 1)           AS avg_ttft_ms,
                   ROUND(AVG(prefill_tps), 1)       AS avg_prefill_tps,
                   ROUND(MIN(prefill_tps), 1)       AS min_prefill_tps,
                   ROUND(MAX(prefill_tps), 1)       AS max_prefill_tps,
                   ROUND(AVG(gen_tps), 1)           AS avg_gen_tps,
                   ROUND(MIN(gen_tps), 1)           AS min_gen_tps,
                   ROUND(MAX(gen_tps), 1)           AS max_gen_tps,
                   SUM(prompt_tokens)               AS total_prompt_tokens,
                   SUM(completion_tokens)           AS total_completion_tokens
            FROM requests WHERE {' AND '.join(where)}
            GROUP BY machine, model ORDER BY machine, model"""
        with self._lock:
            cur = self._db.execute(sql, args)
            col_names = [c[0] for c in cur.description]
            return [dict(zip(col_names, row)) for row in cur.fetchall()]

    def recent(self, limit: int) -> list[RequestRecord]:
        """Returns the most recent request records, ordered newest first."""
        with self._lock:
            cur = self._db.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,))
            col_names = [c[0] for c in cur.description]
            return [self._row_to_record(dict(zip(col_names, row))) for row in cur.fetchall()]

    def query(self, limit: int, offset: int, filters: dict[str, Any]) -> dict[str, Any]:
        """Queries requests with optional column filters; returns rows, total count, and facets."""
        where: list[str] = []
        args: list[Any] = []
        for col in ("model", "machine", "priority", "client", "endpoint"):
            if filters.get(col):
                where.append(f"{col} = ?")
                args.append(filters[col])
        st = filters.get("status")
        if st == "ok":
            where.append("status < 400")
        elif st == "error":
            where.append("status >= 400")
        if filters.get("since"):
            where.append("ts >= ?")
            args.append(float(filters["since"]))
        if filters.get("q"):
            where.append("(request_id LIKE ? OR error LIKE ?)")
            args += [f"%{filters['q']}%"] * 2
        w = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM requests {w}", args).fetchone()[0]
            cur = self._db.execute(
                f"SELECT * FROM requests {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                args + [limit, offset],
            )
            col_names = [c[0] for c in cur.description]
            rows = [self._row_to_record(dict(zip(col_names, r))) for r in cur.fetchall()]
            facets = {
                c: [r[0] for r in self._db.execute(
                    f"SELECT DISTINCT {c} FROM requests WHERE {c} IS NOT NULL ORDER BY {c}"
                )]
                for c in ("model", "machine", "client")
            }
        return {"total": total, "rows": rows, "facets": facets}

    def overview(self, since: float) -> dict[str, Any]:
        """Returns aggregate KPIs (request count, errors, queue wait, throughput) since a timestamp."""
        with self._lock:
            r = self._db.execute("""
                SELECT COUNT(*), SUM(status >= 400), AVG(queue_wait_ms), AVG(gen_tps), AVG(prefill_tps),
                       SUM(COALESCE(prompt_tokens, 0)), SUM(COALESCE(completion_tokens, 0))
                FROM requests WHERE ts >= ?""", (since,)).fetchone()
        keys = ("requests", "errors", "avg_queue_wait_ms", "avg_gen_tps", "avg_prefill_tps",
                "prompt_tokens", "completion_tokens")
        return {k: (round(v, 1) if isinstance(v, float) else (v or 0)) for k, v in zip(keys, r)}

    def close(self) -> None:
        """Closes the SQLite connection."""
        with self._lock:
            self._db.close()
