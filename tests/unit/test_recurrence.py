from __future__ import annotations

from aloepri.attacks.recurrence import frequency_rank_encode


def test_recurrence_encoding_is_substitution_invariant() -> None:
    plain = [7, 2, 7, 9, 2, 7, 5]
    cipher = [40, 11, 40, 3, 11, 40, 99]
    assert frequency_rank_encode(plain) == frequency_rank_encode(cipher)
