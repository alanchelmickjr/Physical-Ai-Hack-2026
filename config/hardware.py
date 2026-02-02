"""Hardware Configuration - Single Source of Truth

This module defines all hardware constants for Johnny Five robots.
Import from here instead of defining motor IDs, ports, or limits elsewhere.

Usage:
    from config.hardware import HardwareConfig

    config = HardwareConfig()
    print(config.LEFT_PORT)  # /dev/ttyACM0
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, List
from enum import Enum


class Bus(Enum):
    """Serial bus identifiers."""
    ACM0 = "/dev/ttyACM0"  # Left side
    ACM1 = "/dev/ttyACM1"  # Right side


@dataclass(frozen=True)
class ServoConfig:
    """Configuration for a single servo."""
    id: int
    name: str
    range: Tuple[float, float]  # min, max degrees or mm
    default: float = 0.0
    servo_type: str = "feetech"


@dataclass(frozen=True)
class SubsystemConfig:
    """Configuration for a subsystem (arm, gantry, etc)."""
    name: str
    bus: Bus
    servos: Tuple[ServoConfig, ...]

    @property
    def ids(self) -> Tuple[int, ...]:
        """Get all servo IDs in this subsystem."""
        return tuple(s.id for s in self.servos)

    @property
    def port(self) -> str:
        """Get the serial port for this subsystem."""
        return self.bus.value


@dataclass
class HardwareConfig:
    """Complete hardware configuration for Johnny Five robots.

    Bus Layout:
        ACM0 (left_port):  Left arm (1-6), wheels (7-9), lift (10)
        ACM1 (right_port): Right arm (1-6), gantry (7-8)

    All motors are Feetech servos via Solo-CLI.
    """

    # ==========================================================================
    # Serial Ports
    # ==========================================================================
    LEFT_PORT: str = "/dev/ttyACM0"
    RIGHT_PORT: str = "/dev/ttyACM1"
    BAUDRATE: int = 1_000_000

    # ==========================================================================
    # Motor IDs - Simple access
    # ==========================================================================
    LEFT_ARM_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    RIGHT_ARM_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    WHEEL_IDS: Tuple[int, ...] = (7, 8, 9)
    LIFT_ID: int = 10
    GANTRY_IDS: Tuple[int, ...] = (7, 8)  # Pan=7, Tilt=8

    # ==========================================================================
    # Joint Limits
    # ==========================================================================
    ARM_LIMITS: Tuple[float, float] = (-150.0, 150.0)
    PAN_LIMITS: Tuple[float, float] = (-90.0, 90.0)
    TILT_LIMITS: Tuple[float, float] = (-45.0, 45.0)
    LIFT_LIMITS: Tuple[float, float] = (0.0, 300.0)  # mm

    # ==========================================================================
    # Default Speeds (0.0 - 1.0)
    # ==========================================================================
    TRACK_SPEED: float = 0.5
    GESTURE_SPEED: float = 0.7
    BASE_SPEED: float = 0.3
    CALIBRATION_SPEED: float = 0.3

    # ==========================================================================
    # Subsystem Configurations (detailed)
    # ==========================================================================
    @property
    def left_arm(self) -> SubsystemConfig:
        """Left SO101 6-DOF arm configuration."""
        return SubsystemConfig(
            name="left_arm",
            bus=Bus.ACM0,
            servos=(
                ServoConfig(1, "shoulder_rotate", (-180, 180), 0),
                ServoConfig(2, "shoulder_pitch", (-90, 90), 0),
                ServoConfig(3, "elbow", (-135, 135), 0),
                ServoConfig(4, "wrist_rotate", (-180, 180), 0),
                ServoConfig(5, "wrist_pitch", (-90, 90), 0),
                ServoConfig(6, "gripper", (0, 100), 0),
            )
        )

    @property
    def right_arm(self) -> SubsystemConfig:
        """Right SO101 6-DOF arm configuration."""
        return SubsystemConfig(
            name="right_arm",
            bus=Bus.ACM1,
            servos=(
                ServoConfig(1, "shoulder_rotate", (-180, 180), 0),
                ServoConfig(2, "shoulder_pitch", (-90, 90), 0),
                ServoConfig(3, "elbow", (-135, 135), 0),
                ServoConfig(4, "wrist_rotate", (-180, 180), 0),
                ServoConfig(5, "wrist_pitch", (-90, 90), 0),
                ServoConfig(6, "gripper", (0, 100), 0),
            )
        )

    @property
    def gantry(self) -> SubsystemConfig:
        """OAK-D Pro 2-DOF camera mount (Feetech pan/tilt)."""
        return SubsystemConfig(
            name="gantry",
            bus=Bus.ACM1,
            servos=(
                ServoConfig(7, "pan", (-90, 90), 0, "feetech"),
                ServoConfig(8, "tilt", (-45, 45), 0, "feetech"),
            )
        )

    @property
    def base(self) -> SubsystemConfig:
        """3-wheel mecanum base."""
        return SubsystemConfig(
            name="base",
            bus=Bus.ACM0,
            servos=(
                ServoConfig(7, "wheel_left", (-360, 360), 0),
                ServoConfig(8, "wheel_right", (-360, 360), 0),
                ServoConfig(9, "wheel_rear", (-360, 360), 0),
            )
        )

    @property
    def lift(self) -> SubsystemConfig:
        """30cm vertical lift."""
        return SubsystemConfig(
            name="lift",
            bus=Bus.ACM0,
            servos=(
                ServoConfig(10, "lift_motor", (0, 300), 0),
            )
        )

    # ==========================================================================
    # Mapping helpers
    # ==========================================================================
    def get_port_for_subsystem(self, subsystem: str) -> str:
        """Get the serial port for a subsystem name."""
        left_subsystems = {"left_arm", "base", "lift", "wheels"}
        right_subsystems = {"right_arm", "gantry"}

        if subsystem in left_subsystems:
            return self.LEFT_PORT
        elif subsystem in right_subsystems:
            return self.RIGHT_PORT
        else:
            raise ValueError(f"Unknown subsystem: {subsystem}")

    def get_subsystem_config(self, subsystem: str) -> SubsystemConfig:
        """Get the SubsystemConfig for a subsystem name."""
        configs = {
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "gantry": self.gantry,
            "base": self.base,
            "lift": self.lift,
        }
        if subsystem not in configs:
            raise ValueError(f"Unknown subsystem: {subsystem}")
        return configs[subsystem]

    @property
    def all_subsystems(self) -> Dict[str, SubsystemConfig]:
        """Get all subsystem configurations."""
        return {
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "gantry": self.gantry,
            "base": self.base,
            "lift": self.lift,
        }

    @property
    def expected_motor_count(self) -> Dict[str, int]:
        """Expected motor count per bus for verification."""
        return {
            "ACM0": 10,  # 6 arm + 3 wheels + 1 lift
            "ACM1": 8,   # 6 arm + 2 gantry
        }


# =============================================================================
# Singleton instance
# =============================================================================
_config: HardwareConfig = None


def get_hardware_config() -> HardwareConfig:
    """Get the singleton hardware configuration."""
    global _config
    if _config is None:
        _config = HardwareConfig()
    return _config
