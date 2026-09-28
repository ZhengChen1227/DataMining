"""漂移协议注册表入口。"""

from __future__ import annotations

from .amplitude import AmplitudeDrift
from .channel import ChannelMask, ChannelPermutation
from .protocol import (
    REGISTRY,
    DriftProtocol,
    NoDrift,
    build_drift,
    distribution_gap,
    list_protocols,
    register,
)
from .temporal import TemporalShift

__all__ = [
    "REGISTRY",
    "DriftProtocol",
    "NoDrift",
    "AmplitudeDrift",
    "ChannelMask",
    "ChannelPermutation",
    "TemporalShift",
    "build_drift",
    "list_protocols",
    "register",
    "distribution_gap",
]