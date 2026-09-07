from __future__ import annotations

from aloepri.product.paths import product_paths


def test_cache_override_does_not_move_state_or_credentials(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    cache = tmp_path / "large-cache"
    monkeypatch.setenv("YINBIAN_HOME", str(home))
    monkeypatch.setenv("YINBIAN_CACHE_DIR", str(cache))
    paths = product_paths()
    assert paths.cache == cache
    assert paths.state == home / "state"
    assert paths.credentials == home / "credentials"


def test_large_data_can_live_outside_application_state(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    data = tmp_path / "model-disk"
    monkeypatch.setenv("YINBIAN_HOME", str(home))
    monkeypatch.setenv("YINBIAN_DATA_DIR", str(data))
    monkeypatch.delenv("YINBIAN_CACHE_DIR", raising=False)

    paths = product_paths()

    assert paths.data == data
    assert paths.source_models == data / "source-models"
    assert paths.private_models == data / "private-models"
    assert paths.evidence == data / "evidence"
    assert paths.cache == data / "cache"
    assert paths.state == home / "state"
    assert paths.credentials == home / "credentials"
