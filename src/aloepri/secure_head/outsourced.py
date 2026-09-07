"""Compatibility imports for the renamed masked Head outsourcing engine.

New code should import :mod:`aloepri.secure_head.masked_outsource_engine` directly.
"""

from aloepri.secure_head.masked_outsource_engine import (
    InProcessGpuHeadWorker,
    MaskedOutsourceHeadEngine,
    QuantizedHead,
)

__all__ = ["InProcessGpuHeadWorker", "MaskedOutsourceHeadEngine", "QuantizedHead"]
