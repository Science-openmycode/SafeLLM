from aloepri.cli import append_optional_noise_seed_arguments, summarize_product_verification


def _generation() -> dict[str, object]:
    return {
        "next_token_equal_after_inverse": True,
        "decode_input_is_tau_plain_next": True,
        "greedy_ids_equal": True,
        "plain_prefill_cache_length": 12,
        "private_prefill_cache_length": 12,
        "decode_cache_length": 13,
    }


def test_no_noise_requires_layerwise_equivalence() -> None:
    result = summarize_product_verification(
        {"alpha_e": 0.0, "alpha_h": 0.0},
        {"overall_pass": True},
        {"all_pass": False, "first_failure": {"operator": "embedding"}},
        _generation(),
    )
    assert result["noise_active"] is False
    assert result["layerwise"]["required"] is True
    assert result["functional_pass"] is False


def test_paper_noise_makes_layerwise_difference_diagnostic() -> None:
    result = summarize_product_verification(
        {"alpha_e": 0.01, "alpha_h": 0.002},
        {"overall_pass": True},
        {"all_pass": False, "first_failure": {"operator": "embedding"}},
        _generation(),
    )
    assert result["noise_active"] is True
    assert result["layerwise"]["required"] is False
    assert result["layerwise"]["role"] == "diagnostic_only"
    assert result["functional_pass"] is True
    assert result["accuracy_gate"] == "external_task_evaluation_required"


def test_noisy_model_treats_greedy_divergence_as_accuracy_evidence() -> None:
    broken_runtime = _generation()
    broken_runtime["greedy_ids_equal"] = False
    broken_runtime["next_token_equal_after_inverse"] = False
    result = summarize_product_verification(
        {"alpha_e": 0.01, "alpha_h": 0.002},
        {"overall_pass": True},
        {"all_pass": False},
        broken_runtime,
    )
    assert result["functional_pass"] is True
    assert result["runtime_required_checks"] == {
        "decode_input_is_tau_plain_next": True,
        "cache_lengths_advance": True,
    }


def test_noisy_model_still_requires_cache_and_coordinate_checks() -> None:
    broken_runtime = _generation()
    broken_runtime["decode_cache_length"] = 99
    result = summarize_product_verification(
        {"alpha_e": 0.01, "alpha_h": 0.002},
        {"overall_pass": True},
        {"all_pass": False},
        broken_runtime,
    )
    assert result["functional_pass"] is False


def test_optional_noise_seeds_are_forwarded_to_converter() -> None:
    command = ["python", "converter.py"]
    append_optional_noise_seed_arguments(
        command,
        {"embedding_noise_seed": 7, "head_noise_seed": 11},
    )
    assert command[-4:] == ["--embedding-noise-seed", "7", "--head-noise-seed", "11"]
