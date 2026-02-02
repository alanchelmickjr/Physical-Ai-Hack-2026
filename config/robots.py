"""Robot Configuration Registry

Defines hardware configurations for different robot platforms.
This enables the same codebase to run on Johnny Five, Booster K1, etc.

Usage:
    from config.robots import get_robot_config, RobotType

    config = get_robot_config(RobotType.BOOSTER_K1)
    print(config.camera_type)  # "zed"
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum


class RobotType(Enum):
    """Supported robot platforms."""
    JOHNNY_FIVE = "johnny_five"
    BOOSTER_K1 = "booster_k1"
    UNITREE_G1 = "unitree_g1"


class CameraType(Enum):
    """Supported camera types."""
    OAK_D = "oak_d"
    OAK_D_PRO = "oak_d_pro"
    ZED = "zed"
    ZED_X = "zed_x"
    ZED_2 = "zed_2"
    REALSENSE_D435 = "realsense_d435"
    REALSENSE_D455 = "realsense_d455"


class MicrophoneType(Enum):
    """Supported microphone array types."""
    RESPEAKER_4MIC = "respeaker_4mic"
    CIRCULAR_6MIC = "circular_6mic"
    UNITREE_4MIC = "unitree_4mic"


class ComputeType(Enum):
    """Supported compute platforms."""
    JETSON_ORIN_8GB = "orin_8gb"
    JETSON_ORIN_NX = "orin_nx"
    JETSON_ORIN_NANO = "orin_nano"


@dataclass
class CameraConfig:
    """Camera hardware configuration."""
    type: CameraType
    width: int = 1920
    height: int = 1080
    fps: int = 30
    hfov: float = 69.0  # Horizontal field of view
    depth_min_mm: int = 100
    depth_max_mm: int = 10000

    # Connection
    interface: str = "usb"  # "usb", "gmsl2"
    device_id: Optional[str] = None


@dataclass
class MicrophoneConfig:
    """Microphone array configuration."""
    type: MicrophoneType
    num_mics: int
    sample_rate: int = 16000
    doa_resolution: float = 10.0  # Degrees

    # USB identifiers
    vendor_id: int = 0
    product_id: int = 0

    # Geometry (for DOA calculation)
    array_radius_mm: float = 0.0
    mic_positions: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class MotorConfig:
    """Motor/actuator configuration."""
    dof: int
    motor_type: str  # "dynamixel", "feetech", "brushless"
    protocol: str = "serial"
    baudrate: int = 1_000_000


@dataclass
class RobotConfig:
    """Complete robot hardware configuration."""
    name: str
    type: RobotType

    # Compute
    compute: ComputeType
    compute_tops: int  # AI compute capability in TOPS

    # Sensors
    camera: CameraConfig
    microphone: MicrophoneConfig

    # Mobility
    mobile_base: bool = True
    base_type: str = "wheeled"  # "wheeled", "bipedal", "tracks"

    # Manipulation
    arms: int = 0
    arm_dof: int = 0
    grippers: bool = False

    # Other
    height_mm: int = 0
    weight_kg: float = 0.0
    battery_minutes: int = 0


# =============================================================================
# Pre-defined Robot Configurations
# =============================================================================

JOHNNY_FIVE_CONFIG = RobotConfig(
    name="Johnny Five",
    type=RobotType.JOHNNY_FIVE,

    # Compute
    compute=ComputeType.JETSON_ORIN_8GB,
    compute_tops=40,

    # Camera - OAK-D Pro on 2-DOF gantry
    camera=CameraConfig(
        type=CameraType.OAK_D_PRO,
        width=1920,
        height=1080,
        fps=30,
        hfov=69.0,
        depth_min_mm=100,
        depth_max_mm=10000,
        interface="usb",
    ),

    # Microphone - ReSpeaker 4-Mic Array
    microphone=MicrophoneConfig(
        type=MicrophoneType.RESPEAKER_4MIC,
        num_mics=4,
        sample_rate=16000,
        doa_resolution=10.0,
        vendor_id=0x2886,
        product_id=0x0018,
        array_radius_mm=32.5,  # ReSpeaker v2.0 radius
        mic_positions=[
            (0, 32.5),    # Front
            (32.5, 0),    # Right
            (0, -32.5),   # Back
            (-32.5, 0),   # Left
        ],
    ),

    # Mobility
    mobile_base=True,
    base_type="wheeled",

    # Manipulation
    arms=2,
    arm_dof=6,
    grippers=True,

    # Physical
    height_mm=600,
    weight_kg=8.0,
    battery_minutes=120,
)


BOOSTER_K1_CONFIG = RobotConfig(
    name="Booster K1",
    type=RobotType.BOOSTER_K1,

    # Compute - Orin NX with 40 TOPS
    compute=ComputeType.JETSON_ORIN_NX,
    compute_tops=40,

    # Camera - ZED camera (user specified)
    camera=CameraConfig(
        type=CameraType.ZED_X,
        width=1920,
        height=1200,
        fps=30,
        hfov=110.0,  # ZED X wide FOV
        depth_min_mm=200,
        depth_max_mm=20000,  # ZED has longer range
        interface="gmsl2",  # ZED X uses GMSL2
    ),

    # Microphone - Circular 6-Mic Array
    microphone=MicrophoneConfig(
        type=MicrophoneType.CIRCULAR_6MIC,
        num_mics=6,
        sample_rate=16000,
        doa_resolution=8.0,  # Better resolution with 6 mics
        vendor_id=0x0000,  # TBD - depends on specific hardware
        product_id=0x0000,
        array_radius_mm=45.0,  # Estimated
        mic_positions=[
            (0, 45.0),      # 0 degrees
            (39.0, 22.5),   # 60 degrees
            (39.0, -22.5),  # 120 degrees
            (0, -45.0),     # 180 degrees
            (-39.0, -22.5), # 240 degrees
            (-39.0, 22.5),  # 300 degrees
        ],
    ),

    # Mobility - Bipedal humanoid
    mobile_base=True,
    base_type="bipedal",

    # Manipulation - 22 DOF total
    arms=2,
    arm_dof=6,
    grippers=True,

    # Physical
    height_mm=950,
    weight_kg=19.5,
    battery_minutes=40,
)


UNITREE_G1_CONFIG = RobotConfig(
    name="Unitree G1",
    type=RobotType.UNITREE_G1,

    # Compute - Jetson Orin with up to 100 TOPS (EDU version)
    compute=ComputeType.JETSON_ORIN_NX,
    compute_tops=100,

    # Camera - Intel RealSense D435
    camera=CameraConfig(
        type=CameraType.REALSENSE_D435,
        width=1280,
        height=720,
        fps=30,
        hfov=87.0,  # RealSense D435 wide FOV
        depth_min_mm=105,
        depth_max_mm=10000,
        interface="usb",
    ),

    # Microphone - Built-in 4-Mic Array
    microphone=MicrophoneConfig(
        type=MicrophoneType.UNITREE_4MIC,
        num_mics=4,
        sample_rate=16000,
        doa_resolution=10.0,
        vendor_id=0x0000,  # Via unitree_sdk2
        product_id=0x0000,
        array_radius_mm=40.0,  # Estimated
        mic_positions=[
            (0, 40.0),    # Front
            (40.0, 0),    # Right
            (0, -40.0),   # Back
            (-40.0, 0),   # Left
        ],
    ),

    # Mobility - Bipedal humanoid (23-43 DOF depending on config)
    mobile_base=True,
    base_type="bipedal",

    # Manipulation - Dex3 hands on EDU version
    arms=2,
    arm_dof=7,  # Per arm
    grippers=True,

    # Physical
    height_mm=1270,  # 127cm
    weight_kg=35.0,
    battery_minutes=120,
)


# =============================================================================
# Registry
# =============================================================================

_ROBOT_CONFIGS: Dict[RobotType, RobotConfig] = {
    RobotType.JOHNNY_FIVE: JOHNNY_FIVE_CONFIG,
    RobotType.BOOSTER_K1: BOOSTER_K1_CONFIG,
    RobotType.UNITREE_G1: UNITREE_G1_CONFIG,
}

_current_robot: Optional[RobotConfig] = None


def get_robot_config(robot_type: Optional[RobotType] = None) -> RobotConfig:
    """Get configuration for a robot type.

    Args:
        robot_type: Robot type to get config for. If None, returns current.

    Returns:
        RobotConfig for the specified robot
    """
    global _current_robot

    if robot_type is None:
        if _current_robot is None:
            # Default to Johnny Five
            _current_robot = JOHNNY_FIVE_CONFIG
        return _current_robot

    if robot_type not in _ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot type: {robot_type}")

    return _ROBOT_CONFIGS[robot_type]


def set_current_robot(robot_type: RobotType) -> RobotConfig:
    """Set the current robot type for the session.

    Args:
        robot_type: Robot type to use

    Returns:
        RobotConfig for the selected robot
    """
    global _current_robot
    _current_robot = get_robot_config(robot_type)
    print(f"[Config] Robot set to: {_current_robot.name}")
    return _current_robot


def register_robot(config: RobotConfig):
    """Register a custom robot configuration.

    Args:
        config: RobotConfig to register
    """
    _ROBOT_CONFIGS[config.type] = config
    print(f"[Config] Registered robot: {config.name}")
