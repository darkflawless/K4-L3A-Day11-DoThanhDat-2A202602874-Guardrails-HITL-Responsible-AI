"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        rid = request_id or f"{user_id}_{len(self.logs)}_{datetime.now(timezone.utc).timestamp()}"
        self._open[rid] = {
            "user_id": user_id,
            "input": text,
            "start_time": datetime.now(timezone.utc).timestamp(),
            "timestamp": utc_now_iso(),
        }
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        start_info = self._open.pop(request_id, None) if request_id else None
        now = datetime.now(timezone.utc).timestamp()
        latency = (
            round(now - start_info["start_time"], 4)
            if start_info and "start_time" in start_info
            else 0.0
        )
        log_entry = {
            "request_id": request_id,
            "timestamp": start_info["timestamp"] if start_info else utc_now_iso(),
            "user_id": user_id,
            "input": start_info["input"] if start_info else "",
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_seconds": latency,
        }
        self.logs.append(log_entry)
        return log_entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
