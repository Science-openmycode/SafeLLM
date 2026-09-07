"""Compatibility imports for the renamed trusted-boundary module.

New code should import :mod:`aloepri.tee.trusted_boundary` directly.
"""

from aloepri.tee.trusted_boundary import (
    GenerationParameters,
    HeadDecision,
    HeadEngine,
    LocalTeeHeadEngine,
    SoftwareTrustedBoundary,
    TrustedBoundary,
    sample_logits,
)

__all__ = [
    "GenerationParameters",
    "HeadDecision",
    "HeadEngine",
    "LocalTeeHeadEngine",
    "SoftwareTrustedBoundary",
    "TrustedBoundary",
    "sample_logits",
]
