"""Configuration module - single source of truth."""

from config.hardware import HardwareConfig, ServoConfig, SubsystemConfig
from config.robots import (
    RobotType,
    RobotConfig,
    CameraType,
    MicrophoneType,
    get_robot_config,
    set_current_robot,
)

__all__ = [
    "HardwareConfig",
    "ServoConfig",
    "SubsystemConfig",
    "RobotType",
    "RobotConfig",
    "CameraType",
    "MicrophoneType",
    "get_robot_config",
    "set_current_robot",
]
