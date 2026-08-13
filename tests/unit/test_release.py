import hashlib
import json

import yaml

from aloepri.release import build_product_release, inspect_product_release


def test_build_release_sanitizes_runtime_and_detects_tampering(tmp_path) -> None:
    server = tmp_path / "server-source"
    server.mkdir()
    (server / "config.json").write_text("{}", encoding="utf-8")
    server_config = server / "config.json"
    (server / "aloepri_manifest.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "files": [
                    {
                        "path": "config.json",
                        "bytes": server_config.stat().st_size,
                        "sha256": hashlib.sha256(server_config.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "product.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_id": "model",
                "key_id": "key",
                "source_model": "secret-source-path",
                "conversion": {"seed": 123},
                "server": {"host": "127.0.0.1", "port": 8000},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "release"
    result = build_product_release(
        output=output, server_package=server, config_path=config
    )
    assert result["pass"] is True
    runtime = (output / "configs" / "runtime.yaml").read_text(encoding="utf-8")
    assert "secret-source-path" not in runtime
    assert "seed" not in runtime
    assert not list(output.rglob("*.safetensors"))
    (output / "README.md").write_text("tampered", encoding="utf-8")
    assert inspect_product_release(output)["pass"] is False
