"""Performance logging: SQLite (for stats) + JSONL (human-readable / ingestible)."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict
from typing import Optional

from .models import RequestRecord
from .repo_request import RequestRepository

log = logging.getLogger("llm_router.metrics")


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
        """Inserts the record via the repository and appends it to the JSONL file."""
        self._repo.insert(rec)
        if self._jsonl:
            with self._jsonl_lock:
                self._jsonl.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                self._jsonl.flush()

    async def record(self, rec: RequestRecord) -> None:
        """Rounds numeric fields, logs the request, and writes it to storage asynchronously."""
        rec.ts = rec.ts or time.time()
        for field in ("queue_wait_ms", "ttft_ms", "total_ms", "prefill_tps", "gen_tps",
                      "total_duration_request"):
            setattr(rec, field, _r(getattr(rec, field)))
        def fmt(v, unit=""):
            return "-" if v is None else f"{v}{unit}"
        log.info(
            "%s %s -> %s [%s] status=%s wait=%s ttft=%s total=%s prompt=%s compl=%s "
            "prefill=%s gen=%s (%s) duration=%s%s",
            rec.endpoint, rec.model, fmt(rec.machine), rec.priority, rec.status,
            fmt(rec.queue_wait_ms, "ms"), fmt(rec.ttft_ms, "ms"), fmt(rec.total_ms, "ms"),
            fmt(rec.prompt_tokens), fmt(rec.completion_tokens),
            fmt(rec.prefill_tps, " tok/s"), fmt(rec.gen_tps, " tok/s"),
            fmt(rec.metrics_source), fmt(rec.total_duration_request, "ms"),
            f" | {rec.error}" if rec.error else "",
        )
        try:
            await asyncio.to_thread(self._write, rec)
        except Exception:  # logging must never fail a request
            log.exception("Failed to write metrics")

    async def query(self, limit: int = 50, offset: int = 0, **filters) -> dict:
        """Async wrapper for repository query."""
        return await asyncio.to_thread(self._repo.query, limit, offset, filters)

    async def overview(self, hours: float = 1) -> dict:
        """Async wrapper for repository overview, defaulting to the last hour."""
        return await asyncio.to_thread(self._repo.overview, time.time() - hours * 3600)

    async def stats(
        self, hours: float = 24, model: Optional[str] = None, machine: Optional[str] = None
    ):
        """Async wrapper for repository stats."""
        return await asyncio.to_thread(
            self._repo.stats, time.time() - hours * 3600, model, machine
        )

    async def recent(self, limit: int = 50):
        """Async wrapper for repository recent."""
        return await asyncio.to_thread(self._repo.recent, limit)

    def close(self) -> None:
        """Closes the repository's SQLite connection and the JSONL file."""
        self._repo.close()
        if self._jsonl:
            with self._jsonl_lock:
                self._jsonl.close()
