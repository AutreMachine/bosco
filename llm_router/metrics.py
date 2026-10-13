"""Performance logging: SQLite (for stats) + JSONL (human-readable / ingestible)."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

from .repo_request import RequestRepository

log = logging.getLogger("llm_router.metrics")


@dataclass
class RequestRecord:
    request_id: str
    endpoint: str
    model: str
    machine: Optional[str]
    priority: str
    stream: bool
    status: int
    queue_wait_ms: float = 0.0
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    prefill_tps: Optional[float] = None
    gen_tps: Optional[float] = None
    metrics_source: Optional[str] = None   # "measured" | "backend_timings" | "total_only"
    error: Optional[str] = None
    client: Optional[str] = None          # name of the API key used
    client_ip: Optional[str] = None
    ts: float = 0.0


def _r(x: Optional[float], n: int = 2) -> Optional[float]:
    """Rounds a float to n decimal places, returning None if the value is None."""
    return None if x is None else round(x, n)


class MetricsStore:
    def __init__(self, db_path: str, jsonl_path: Optional[str] = None):
        """Creates the request repository and opens the optional JSONL log file."""
        self._repo = RequestRepository(db_path)
        self._jsonl_lock = threading.Lock()
        self._jsonl = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None

    def _write(self, rec: RequestRecord) -> None:
        """Serializes the record, inserts it via the repository, and appends to the JSONL file."""
        row = asdict(rec)
        self._repo.insert(row)
        if self._jsonl:
            with self._jsonl_lock:
                self._jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._jsonl.flush()

    async def record(self, rec: RequestRecord) -> None:
        """Rounds numeric fields, logs the request, and writes it to storage asynchronously."""
        rec.ts = rec.ts or time.time()
        for f in ("queue_wait_ms", "ttft_ms", "total_ms", "prefill_tps", "gen_tps"):
            setattr(rec, f, _r(getattr(rec, f)))
        def f(v, unit=""):
            return "-" if v is None else f"{v}{unit}"
        log.info(
            "%s %s -> %s [%s] status=%s file=%s ttft=%s total=%s prompt=%s compl=%s "
            "prefill=%s gen=%s (%s)%s",
            rec.endpoint, rec.model, f(rec.machine), rec.priority, rec.status, f(rec.queue_wait_ms, "ms"),
            f(rec.ttft_ms, "ms"), f(rec.total_ms, "ms"), f(rec.prompt_tokens), f(rec.completion_tokens),
            f(rec.prefill_tps, " tok/s"), f(rec.gen_tps, " tok/s"), f(rec.metrics_source),
            f" | {rec.error}" if rec.error else "",
        )
        try:
            await asyncio.to_thread(self._write, rec)
        except Exception:  # la journalisation ne doit jamais faire échouer une requête
            log.exception("Failed to write metrics")

    async def query(self, limit: int = 50, offset: int = 0, **filters) -> dict:
        """Async wrapper for _query."""
        return await asyncio.to_thread(self._repo.query, limit, offset, filters)

    async def overview(self, hours: float = 1) -> dict:
        """Async wrapper for _overview, defaulting to the last hour."""
        return await asyncio.to_thread(self._repo.overview, time.time() - hours * 3600)

    async def stats(self, hours: float = 24, model: Optional[str] = None, machine: Optional[str] = None):
        """Async wrapper for _stats."""
        return await asyncio.to_thread(self._repo.stats, time.time() - hours * 3600, model, machine)

    async def recent(self, limit: int = 50):
        """Async wrapper for _recent."""
        return await asyncio.to_thread(self._repo.recent, limit)

    def close(self) -> None:
        """Closes the repository's SQLite connection and the JSONL file."""
        self._repo.close()
        if self._jsonl:
            with self._jsonl_lock:
                self._jsonl.close()
