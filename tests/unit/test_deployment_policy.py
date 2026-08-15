from __future__ import annotations

import pytest

from aloepri.product.deployment_policy import require_validated_hf_deployment


def test_only_checkpoint_validated_catalog_revision_can_enter_hf_deployer() -> None:
    require_validated_hf_deployment(
        {
            "source": {
                "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            }
        }
    )


def test_family_compatibility_does_not_count_as_deployment_acceptance() -> None:
    with pytest.raises(ValueError, match="checkpoint-specific"):
        require_validated_hf_deployment(
            {
                "source": {
                    "repo_id": "Qwen/Qwen3-0.6B",
                    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                }
            }
        )


def test_validated_model_rejects_unvalidated_revision() -> None:
    with pytest.raises(ValueError, match="revision"):
        require_validated_hf_deployment(
            {
                "source": {
                    "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "revision": "moving-main",
                }
            }
        )
