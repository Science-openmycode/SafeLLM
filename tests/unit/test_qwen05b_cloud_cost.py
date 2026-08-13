from decimal import Decimal

from scripts.calculate_qwen05b_cloud_cost import calculate_scenario, ceil_to_increment


def test_ceil_to_billing_increment() -> None:
    assert ceil_to_increment(Decimal("48.6"), Decimal("1.0")) == Decimal("49.0")


def test_calculate_scenario_uses_gpu_hours() -> None:
    result = calculate_scenario(
        {
            "description": "test",
            "hourly_rate": "2.00",
            "gpu_count": 2,
            "billing_increment_hours": "1.0",
            "contingency_percent": "20",
            "stages": {"run": "10.0"},
        }
    )
    assert result["reserved_instance_hours"] == "12.0"
    assert result["reserved_gpu_hours"] == "24.0"
    assert result["total_cost"] == "48.00"
