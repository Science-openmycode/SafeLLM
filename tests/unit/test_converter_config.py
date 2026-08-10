from types import SimpleNamespace

from scripts.convert_paper_qwen2_checkpoint import (
    get_rope_theta,
    paper_alignment_profile,
    public_metadata,
    resolve_noise_seeds,
)


def test_get_rope_theta_from_legacy_field() -> None:
    assert get_rope_theta(SimpleNamespace(rope_theta=1_000_000.0)) == 1_000_000.0


def test_get_rope_theta_from_transformers5_mapping() -> None:
    config = SimpleNamespace(rope_parameters={"rope_theta": 1_000_000.0})
    assert get_rope_theta(config) == 1_000_000.0


def test_get_rope_theta_default() -> None:
    assert get_rope_theta(SimpleNamespace()) == 10_000.0


def test_noise_seed_resolution_preserves_defaults_and_accepts_independent_draws() -> None:
    assert resolve_noise_seeds(100, None, None) == (102, 103)
    assert resolve_noise_seeds(100, 7, 11) == (7, 11)


def test_public_metadata_removes_reconstructable_seeds() -> None:
    metadata = {
        "model_id": "private-qwen05b",
        "seed": 1,
        "embedding_noise_seed": 2,
        "head_noise_seed": 3,
        "lambda": 0.3,
    }
    assert public_metadata(metadata) == {"model_id": "private-qwen05b", "lambda": 0.3}


def test_paper_alignment_profile_discloses_corrected_functional_settings() -> None:
    args = SimpleNamespace(
        rms_mode="exact-metric",
        block_beta=1,
        rope_frequency_mode="qwen-actual",
        uvo_condition_max=42.0,
        attention_compute_dtype="float64",
        alpha_e=0.0,
        alpha_h=0.0,
    )

    assert paper_alignment_profile(args) == {
        "algorithm1": "shape-corrected-nullspaces",
        "rmsnorm": "corrected-exact-derived-metric",
        "blockperm": "disabled-beta-one-due-noncommuting-rope",
        "rope_frequencies": "architecture-correct-qwen",
        "uvo_sampling": "conditioned-paper-gaussian-for-numerical-stability",
        "attention_precision": "float64",
        "embedding_noise_active": False,
        "head_noise_active": False,
    }
