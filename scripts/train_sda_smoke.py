from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from aloepri.attacks.corpus import load_corpus_texts
from aloepri.attacks.metrics import corpus_bleu4
from aloepri.attacks.recurrence import frequency_rank_encode
from aloepri.evidence import run_provenance


class RecurrenceDecoder(nn.Module):
    def __init__(self, rank_vocab: int, output_vocab: int, max_len: int) -> None:
        super().__init__()
        width = 128
        self.rank = nn.Embedding(rank_vocab, width, padding_idx=0)
        self.position = nn.Embedding(max_len, width)
        layer = nn.TransformerEncoderLayer(
            width, 8, dim_feedforward=256, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.output = nn.Linear(width, output_vocab)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(values.shape[1], device=values.device)
        hidden = self.rank(values) + self.position(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            values.shape[1], device=values.device
        )
        return self.output(self.transformer(hidden, mask=causal_mask))


def pad(sequences: list[list[int]], length: int, value: int) -> torch.Tensor:
    return torch.tensor([item[:length] + [value] * (length - len(item)) for item in sequences])


def main() -> None:
    parser = argparse.ArgumentParser(description="SDA recurrence-decoder smoke experiment")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--train-corpus", type=Path, action="append")
    parser.add_argument("--test-corpus", type=Path, action="append")
    parser.add_argument(
        "--knowledge-setting",
        choices=("unrelated", "domain-aware", "distribution-aware"),
        default="distribution-aware",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.prompts is not None:
        if args.train_corpus or args.test_corpus:
            parser.error("--prompts cannot be combined with explicit corpus arguments")
        source_paths = [args.prompts]
        train_texts = load_corpus_texts(source_paths)
        test_texts: list[str] | None = None
    else:
        if not args.train_corpus or not args.test_corpus:
            parser.error("provide --prompts or both --train-corpus and --test-corpus")
        source_paths = [*args.train_corpus, *args.test_corpus]
        train_texts = load_corpus_texts(args.train_corpus)
        test_texts = load_corpus_texts(args.test_corpus)
        if args.knowledge_setting != "distribution-aware" and {
            path.resolve() for path in args.train_corpus
        }.intersection(path.resolve() for path in args.test_corpus):
            parser.error("unrelated/domain-aware SDA requires disjoint train and test files")
    torch.manual_seed(20260803)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    sequences = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"][: args.max_len]
        for prompt in train_texts
    ]
    sequences = [sequence for sequence in sequences if sequence]
    permutation = torch.randperm(len(sequences), generator=torch.Generator().manual_seed(20260803))
    sequences = [sequences[index] for index in permutation.tolist()]
    if test_texts is None:
        split = max(1, int(len(sequences) * 0.8))
        train_plain, test_plain = sequences[:split], sequences[split:]
    else:
        train_plain = sequences
        test_plain = [
            tokenizer(prompt, add_special_tokens=False)["input_ids"][: args.max_len]
            for prompt in test_texts
        ]
        test_plain = [sequence for sequence in test_plain if sequence]
    if args.train_limit:
        train_plain = train_plain[: args.train_limit]
    if args.test_limit:
        test_plain = test_plain[: args.test_limit]
    if not train_plain or not test_plain:
        raise ValueError("SDA requires non-empty, disjoint train and test sequences")
    max_len = min(args.max_len, max(map(len, train_plain + test_plain)))
    train_rank = pad([frequency_rank_encode(item) for item in train_plain], max_len, 0)
    test_rank = pad([frequency_rank_encode(item) for item in test_plain], max_len, 0)
    train_target = pad(train_plain, max_len, -100)
    test_target = pad(test_plain, max_len, -100)
    rank_vocab = max(int(train_rank.max()), int(test_rank.max())) + 1
    model = RecurrenceDecoder(rank_vocab, tokenizer.vocab_size, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(20260804)
    for _ in range(args.steps):
        indices = torch.randint(0, len(train_rank), (args.batch_size,), generator=generator)
        rank_batch = train_rank[indices].to(device)
        target_batch = train_target[indices].to(device)
        logits = model(rank_batch)
        loss = nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(test_rank), args.batch_size):
            batch = test_rank[start : start + args.batch_size].to(device)
            predictions.append(model(batch).argmax(-1).cpu())
    prediction = torch.cat(predictions)
    valid = test_target != -100
    token_accuracy = float((prediction[valid] == test_target[valid]).float().mean())
    recovered = [
        tokenizer.decode(prediction[index][valid[index]].tolist(), skip_special_tokens=True)
        for index in range(len(test_plain))
    ]
    references = [target[target != -100].tolist() for target in test_target]
    hypotheses = [
        prediction[index][valid[index]].tolist() for index in range(len(test_plain))
    ]
    bleu4 = corpus_bleu4(references, hypotheses)
    payload = {
        "attack": "SDA-recurrence-causal-transformer",
        "scope": "recurrence_decoder_held_out_corpus",
        "knowledge_setting": args.knowledge_setting,
        "train_corpus_files": [
            str(path.resolve()) for path in (args.train_corpus or source_paths)
        ],
        "test_corpus_files": [
            str(path.resolve()) for path in (args.test_corpus or source_paths)
        ],
        "experiment_scale": (
            "formal_10000_train_2000_test"
            if len(train_plain) >= 10000 and len(test_plain) >= 2000
            else "smoke"
        ),
        "paper_exact_scale_claimed": False,
        "true_tau_used_for_training": False,
        "train_sequences": len(train_plain),
        "test_sequences": len(test_plain),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_length": max_len,
        "training_token_presentations": args.steps * args.batch_size * max_len,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "token_accuracy": token_accuracy,
        "bleu4": bleu4,
        "bleu4_implementation": "aloepri_corpus_clipped_ngrams_nltk_method1_compatible",
        "recovered_text_sample": recovered[:20],
        "provenance": run_provenance(
            script=Path(__file__),
            original=args.tokenizer,
            data_files=source_paths,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
