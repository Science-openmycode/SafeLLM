from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Literal

import psutil
import torch


@dataclass(frozen=True)
class ResourceObservation:
    peak_rss_gib: float
    peak_gpu_allocated_gib: float
    gpu_free_before_gib: float | None
    gpu_budget_gib: float
    host_budget_gib: float

    @property
    def pass_(self) -> bool:
        return (
            self.peak_rss_gib <= self.host_budget_gib
            and self.peak_gpu_allocated_gib <= self.gpu_budget_gib
        )


class ResourceBudgetMonitor:
    def __init__(
        self,
        *,
        gpu_budget_gib: float = 4.5,
        minimum_free_gpu_gib: float = 1.2,
        host_budget_gib: float = 11.0,
        sample_seconds: float = 0.05,
    ) -> None:
        self.gpu_budget_gib = gpu_budget_gib
        self.minimum_free_gpu_gib = minimum_free_gpu_gib
        self.host_budget_gib = host_budget_gib
        self.sample_seconds = sample_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0
        self._gpu_free_before: float | None = None
        self.observation: ResourceObservation | None = None

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            self._peak_rss = max(self._peak_rss, int(process.memory_info().rss))
            self._stop.wait(self.sample_seconds)

    def __enter__(self) -> ResourceBudgetMonitor:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            self._gpu_free_before = free / 1024**3
            if self._gpu_free_before < self.minimum_free_gpu_gib:
                raise RuntimeError(
                    f"free GPU memory {self._gpu_free_before:.2f} GiB is below "
                    f"required {self.minimum_free_gpu_gib:.2f} GiB"
                )
            torch.cuda.reset_peak_memory_stats()
        self._sample_once()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample_once(self) -> None:
        self._peak_rss = max(self._peak_rss, int(psutil.Process().memory_info().rss))

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exception, traceback
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_seconds * 4))
        self._sample_once()
        gpu_peak = (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        )
        self.observation = ResourceObservation(
            self._peak_rss / 1024**3,
            gpu_peak,
            self._gpu_free_before,
            self.gpu_budget_gib,
            self.host_budget_gib,
        )
        if exception_type is None and not self.observation.pass_:
            raise RuntimeError(
                "conversion exceeded resource budget: "
                f"RSS={self.observation.peak_rss_gib:.2f}/{self.host_budget_gib:.2f} GiB, "
                f"GPU={gpu_peak:.2f}/{self.gpu_budget_gib:.2f} GiB"
            )
        return False


def wait_for_minimum_free_gpu(minimum_free_gib: float, *, timeout_seconds: float = 0.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if not torch.cuda.is_available():
            return True
        free, _ = torch.cuda.mem_get_info()
        if free / 1024**3 >= minimum_free_gib:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
