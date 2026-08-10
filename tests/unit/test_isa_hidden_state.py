import torch

from scripts.run_isa_hidden_state import nearest_tokens, token_gram


def test_token_gram_is_dimension_invariant_for_orthogonal_map() -> None:
    hidden = torch.randn(5, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    torch.testing.assert_close(token_gram(hidden), token_gram(hidden @ q), atol=1e-5, rtol=1e-5)


def test_nearest_tokens_returns_embedding_row_ids() -> None:
    embedding = torch.eye(4)
    candidates = torch.stack((embedding[2], embedding[0]))
    assert nearest_tokens(candidates, embedding).tolist() == [2, 0]
