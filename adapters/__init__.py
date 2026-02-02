# Robot Adapters
#
# These adapters provide a unified interface for robot control,
# allowing the same tools to work across different hardware platforms.
#
# Supported robots:
# - Johnny Five (OAK-D, ReSpeaker 4-Mic, Feetech servos)
# - Booster K1 (ZED X, 6-Mic array, ROS2 bipedal)
#
# To add a new robot, create an adapter in this directory.

from .base import RobotAdapter, Subsystem, ActionResult, ActionPrimitive, MockAdapter
from .johnny5 import Johnny5Adapter
from .booster_k1 import BoosterK1Adapter

__all__ = [
    "RobotAdapter",
    "Subsystem",
    "ActionResult",
    "ActionPrimitive",
    "MockAdapter",
    "Johnny5Adapter",
    "BoosterK1Adapter",
]
