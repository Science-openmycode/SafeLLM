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
