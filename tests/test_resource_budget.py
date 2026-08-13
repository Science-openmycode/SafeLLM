from __future__ import annotations

from aloepri.resources import ResourceBudgetMonitor, wait_for_minimum_free_gpu


def test_resource_monitor_records_cpu_process_and_budget() -> None:
    with ResourceBudgetMonitor(gpu_budget_gib=100, host_budget_gib=100) as monitor:
        payload = bytearray(1024 * 1024)
        assert len(payload) == 1024 * 1024
    assert monitor.observation is not None
    assert monitor.observation.peak_rss_gib > 0
    assert monitor.observation.pass_ is True


def test_wait_for_gpu_treats_cpu_runtime_as_available() -> None:
    assert wait_for_minimum_free_gpu(1.2, timeout_seconds=0) is True
