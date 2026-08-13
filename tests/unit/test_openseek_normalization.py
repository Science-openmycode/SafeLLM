from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer

from aloepri.conversion.openseek import _build_fast_tokenizer


def test_openseek_fast_tokenizer_matches_upstream_slow_tokenizer(tmp_path: Path) -> None:
    source = Path("data/models/openseek-small-v1-sft").resolve()
    if not (source / "qwen.tiktoken").is_file():
        return
    sys.path.insert(0, str(source))
    try:
        from tokenization_qwen import QWenTokenizer

        slow = QWenTokenizer.from_pretrained(source, local_files_only=True)
    finally:
        sys.path.remove(str(source))
    output = tmp_path / "tokenizer"
    output.mkdir()
    _build_fast_tokenizer(source, output)
    fast = AutoTokenizer.from_pretrained(output, local_files_only=True)
    samples = (
        "你好，请用一句话介绍你自己。",
        "Hello, world!",
        "代码：def f(x): return x + 1",
        "空格  tabs\t换行\n第二行",
    )
    for text in samples:
        slow_ids = slow.encode(text, allowed_special="all")
        fast_ids = fast.encode(text, add_special_tokens=False)
        assert fast_ids == slow_ids
        assert fast.decode(fast_ids) == text
    assert len(fast) == 151851
    assert fast.bos_token_id == 151644
    assert fast.eos_token_id == 151645
    assert fast.pad_token_id == 151643
