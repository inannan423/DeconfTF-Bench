#!/usr/bin/env python3
"""Shared stop rules for the exploratory E1–E4 runs. Does not call a model."""
from __future__ import annotations

import json
import threading
from pathlib import Path

PARSE_FAIL_RATE = 0.20
PARSE_FAIL_MIN_N = 20
STAGE_CAP_FACTOR = 1.5


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def stage_spent(rows: list) -> float:
    total = 0.0
    for row in rows:
        entry = row.get("ledger_entry") or {}
        total += float(entry.get("cost_cny") or 0.0)
    return total


class StageGuard:
    """Stops a stage when spend passes the cap, or parsing fails.

The cap is 1.5 times the estimate unless ``cap_cny`` is set. E5 sets it to 175.
"""

    def __init__(self, estimate_cny: float, already_spent: float = 0.0, cap_cny: float | None = None) -> None:
        self.cap = estimate_cny * STAGE_CAP_FACTOR if cap_cny is None else float(cap_cny)
        self.spent = already_spent
        self.reserved = 0.0
        self.done = 0
        self.failed = 0
        self.reason = ""
        self._lock = threading.Lock()

    def reserve(self, est_cost: float) -> str:
        with self._lock:
            if self.reason:
                return self.reason
            if self.spent + self.reserved + est_cost > self.cap:
                self.reason = (
                    f"stage stop: spent {self.spent:.4f} + reserved {self.reserved:.4f} "
                    f"+ preflight {est_cost:.4f} > {self.cap:.2f} (1.5 times the estimate)"
                )
                return self.reason
            self.reserved += est_cost
            return ""

    def settle(self, est_cost: float, actual_cost: float, parsed: bool, track_parse: bool = True) -> str:
        with self._lock:
            self.reserved = max(0.0, self.reserved - est_cost)
            self.spent += actual_cost
            if track_parse:
                self.done += 1
                if not parsed:
                    self.failed += 1
            if track_parse and self.done >= PARSE_FAIL_MIN_N and self.failed / self.done > PARSE_FAIL_RATE:
                self.reason = (
                    f"stage stop: parse failures {self.failed}/{self.done} "
                    f"({100 * self.failed / self.done:.1f}%)"
                )
            elif self.spent > self.cap:
                self.reason = f"stage stop: spent {self.spent:.4f} > {self.cap:.2f}"
            return self.reason
