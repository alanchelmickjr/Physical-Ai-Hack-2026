"""Robot Factory

Unified entry point for creating robot instances with all components.
Handles adapter, camera, and microphone setup based on robot type.

Usage:
    from robot_factory import create_robot, RobotType

    # Create a Booster K1
    robot = create_robot(RobotType.BOOSTER_K1)
    await robot.adapter.connect()
    robot.camera.start()
    robot.microphone.start()

    # Or detect automatically
    robot = create_robot()  # Auto-detects hardware
"""

from dataclasses import dataclass
from typing import Optional

from config.robots import (
    RobotType,
    RobotConfig,
    CameraType,
    MicrophoneType,
    get_robot_config,
    set_current_robot,
)
from adapters.base import RobotAdapter
from cameras.base import SpatialCamera
from microphones.base import MicrophoneArray


@dataclass
class Robot:
    """Complete robot instance with all components."""
    config: RobotConfig
    adapter: RobotAdapter
    camera: Optional[SpatialCamera] = None
    microphone: Optional[MicrophoneArray] = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def robot_type(self) -> RobotType:
        return self.config.type

    def start_sensors(self) -> bool:
        """Start camera and microphone.

        Returns:
            True if all sensors started successfully
        """
        success = True

        if self.camera:
            if not self.camera.start():
                print(f"[{self.name}] Camera failed to start")
                success = False

        if self.microphone:
            if not self.microphone.start():
                print(f"[{self.name}] Microphone failed to start")
                success = False

        return success

    def stop_sensors(self):
        """Stop camera and microphone."""
        if self.camera:
            self.camera.stop()
        if self.microphone:
            self.microphone.stop()


def create_robot(
    robot_type: Optional[RobotType] = None,
    skip_camera: bool = False,
    skip_microphone: bool = False,
) -> Robot:
    """Create a robot instance with all components.

    Args:
        robot_type: Type of robot to create. If None, auto-detects.
        skip_camera: Don't initialize camera
        skip_microphone: Don't initialize microphone

    Returns:
        Robot instance with adapter, camera, and microphone
    """
    # Auto-detect or use specified type
    if robot_type is None:
        robot_type = _detect_robot_type()

    # Get configuration
    config = get_robot_config(robot_type)
    set_current_robot(robot_type)

    # Create adapter
    adapter = _create_adapter(robot_type)

    # Create camera
    camera = None
    if not skip_camera:
        camera = _create_camera(config.camera.type)

    # Create microphone
    microphone = None
    if not skip_microphone:
        microphone = _create_microphone(config.microphone.type)

    robot = Robot(
        config=config,
        adapter=adapter,
        camera=camera,
        microphone=microphone,
    )

    print(f"[RobotFactory] Created {config.name}")
    print(f"  Camera: {config.camera.type.value if camera else 'disabled'}")
    print(f"  Microphone: {config.microphone.type.value if microphone else 'disabled'}")
    print(f"  Compute: {config.compute.value} ({config.compute_tops} TOPS)")

    return robot


def _detect_robot_type() -> RobotType:
    """Auto-detect robot type based on connected hardware.

    Returns:
        Detected RobotType (defaults to JOHNNY_FIVE)
    """
    # Try to detect ZED camera (indicates Booster K1)
    try:
        import pyzed.sl as sl
        zed = sl.Camera()
        init = sl.InitParameters()
        if zed.open(init) == sl.ERROR_CODE.SUCCESS:
            zed.close()
            print("[RobotFactory] Detected ZED camera -> Booster K1")
            return RobotType.BOOSTER_K1
    except:
        pass

    # Try to detect OAK-D (indicates Johnny Five)
    try:
        import depthai as dai
        devices = dai.Device.getAllAvailableDevices()
        if devices:
            print("[RobotFactory] Detected OAK-D camera -> Johnny Five")
            return RobotType.JOHNNY_FIVE
    except:
        pass

    # Default to Johnny Five
    print("[RobotFactory] No camera detected, defaulting to Johnny Five")
    return RobotType.JOHNNY_FIVE


def _create_adapter(robot_type: RobotType) -> RobotAdapter:
    """Create the appropriate adapter for a robot type."""
    if robot_type == RobotType.JOHNNY_FIVE:
        from adapters.johnny5 import Johnny5Adapter
        return Johnny5Adapter()

    elif robot_type == RobotType.BOOSTER_K1:
        from adapters.booster_k1 import BoosterK1Adapter
        return BoosterK1Adapter()

    else:
        raise ValueError(f"Unknown robot type: {robot_type}")


def _create_camera(camera_type: CameraType) -> Optional[SpatialCamera]:
    """Create the appropriate camera for a camera type."""
    try:
        from cameras.factory import get_camera
        return get_camera(camera_type)
    except Exception as e:
        print(f"[RobotFactory] Camera init failed: {e}")
        return None


def _create_microphone(mic_type: MicrophoneType) -> Optional[MicrophoneArray]:
    """Create the appropriate microphone for a mic type."""
    try:
        from microphones.factory import get_microphone
        return get_microphone(mic_type)
    except Exception as e:
        print(f"[RobotFactory] Microphone init failed: {e}")
        return None


# Convenience exports
__all__ = ['create_robot', 'Robot', 'RobotType']
