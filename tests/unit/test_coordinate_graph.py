from __future__ import annotations

from aloepri.keys.coordinate_graph import qwen2_residual_boundaries, verify_global_residual_key


def test_qwen05b_global_residual_graph() -> None:
    boundaries = qwen2_residual_boundaries(plain_dim=896, expansion_h=128, num_layers=24)
    assert len(boundaries) == 50
    assert all(boundary.private_dim == 1152 for boundary in boundaries)
    verify_global_residual_key(boundaries)
