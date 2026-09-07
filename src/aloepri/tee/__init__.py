"""Trusted-boundary support for the additive ``tee_gm`` product mode.

The existing permutation runtime intentionally does not import this package.
That keeps old checkpoints and their hot path independent from TEE support.
"""

from aloepri.tee.config import BoundaryMode, SecurityMode, SecurityProfile, TeeBackend

__all__ = ["BoundaryMode", "SecurityMode", "SecurityProfile", "TeeBackend"]
