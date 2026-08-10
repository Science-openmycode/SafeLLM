from __future__ import annotations


def placeholder_metric(references: list[str], predictions: list[list[str]]) -> float:
    del references, predictions
    return 0.0


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + response for response in resp]
        for resp, doc in zip(resps, docs, strict=True)
    ]
