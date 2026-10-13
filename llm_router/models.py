"""Shared data models used across the router."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestRecord:
    """One completed (or failed) request, as stored in metrics."""
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
    request_time: Optional[float] = None   # UTC Unix timestamp supplied by the caller (X-Request-Time)
    total_duration_request: Optional[float] = None  # ms from request_time to response complete
