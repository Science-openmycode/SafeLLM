from __future__ import annotations

import threading
from pathlib import Path

from aloepri.conversion.executor import (
    ConversionCancelled,
    ConversionPaused,
    execute_conversion_plan,
)
from aloepri.jobs.store import JobStore
from aloepri.planning import ConversionPlan


class ConversionWorker:
    """Small in-process worker used by Studio; durable state remains in SQLite."""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def start(self, plan: ConversionPlan) -> None:
        with self._lock:
            active = self._threads.get(plan.job_id)
            if active is not None and active.is_alive():
                raise ValueError(f"job already has an active worker: {plan.job_id}")
            thread = threading.Thread(
                target=self._run,
                args=(plan,),
                name=f"aloepri-{plan.job_id[:8]}",
                daemon=True,
            )
            self._threads[plan.job_id] = thread
            thread.start()

    def _run(self, plan: ConversionPlan) -> None:
        try:
            execute_conversion_plan(plan, self.store)
        except (ConversionPaused, ConversionCancelled):
            pass
        finally:
            with self._lock:
                self._threads.pop(plan.job_id, None)

    def start_from_path(self, path: Path) -> str:
        plan = ConversionPlan.load(path)
        self.start(plan)
        return plan.job_id

    def active(self, job_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(job_id)
            return thread is not None and thread.is_alive()
