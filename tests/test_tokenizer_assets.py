from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from aloepri.planning import build_catalog_plan
from aloepri.product.tokenizer_assets import materialize_local_tokenizer


def _write_fast_tokenizer(root: Path) -> None:
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<bos>": 1, "<eos>": 2, "hello": 3},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    root.mkdir(parents=True)
    tokenizer.save_pretrained(root)


def test_materialize_prefers_converted_self_contained_tokenizer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "MissingRepositoryTokenizer",
                "auto_map": {"AutoTokenizer": ["missing.Tokenizer", None]},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "private"
    _write_fast_tokenizer(output)
    (output / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message['content'] }}{% endfor %}",
        encoding="utf-8",
    )

    plan = build_catalog_plan("qwen2.5-0.5b-instruct", output_uri=str(output))
    plan.source["cache_path"] = str(source)
    destination = materialize_local_tokenizer(
        plan, destination_root=tmp_path / "client-tokenizers"
    )

    loaded = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    assert loaded.encode("hello", add_special_tokens=False) == [3]
    manifest = json.loads((destination / "yinbian_tokenizer.json").read_text())
    assert Path(manifest["source"]) == output.resolve()
    assert (destination / "chat_template.jinja").is_file()
    assert materialize_local_tokenizer(
        plan, destination_root=tmp_path / "client-tokenizers"
    ) == destination
