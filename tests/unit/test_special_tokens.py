from types import SimpleNamespace

import torch

from aloepri.conversion.vocab_checkpoint import _map_special_tokens


def test_special_token_ids_are_mapped() -> None:
    config = SimpleNamespace(bos_token_id=1, eos_token_id=[2, 3], pad_token_id=None)
    tau = torch.tensor([2, 0, 3, 1])
    _map_special_tokens(config, tau)
    assert config.bos_token_id == 0
    assert config.eos_token_id == [3, 1]
    assert config.pad_token_id is None
