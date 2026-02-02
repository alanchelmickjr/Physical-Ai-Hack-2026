"""Camera Factory

Creates the appropriate camera instance based on type.
Register custom cameras here.
"""

from typing import Dict, Type, Optional

from .base import SpatialCamera, CameraType


# Camera registry
_CAMERA_REGISTRY: Dict[CameraType, Type[SpatialCamera]] = {}


def register_camera(camera_type: CameraType, camera_class: Type[SpatialCamera]):
    """Register a camera implementation.

    Args:
        camera_type: CameraType enum value
        camera_class: SpatialCamera subclass
    """
    _CAMERA_REGISTRY[camera_type] = camera_class
    print(f"[CameraFactory] Registered {camera_type.value}: {camera_class.__name__}")


def get_camera(camera_type: CameraType) -> SpatialCamera:
    """Get a camera instance for the given type.

    Args:
        camera_type: Type of camera to create

    Returns:
        SpatialCamera instance

    Raises:
        ValueError: If camera type not registered
    """
    # Lazy load implementations
    _ensure_registered()

    if camera_type not in _CAMERA_REGISTRY:
        raise ValueError(f"Unknown camera type: {camera_type}. "
                        f"Available: {list(_CAMERA_REGISTRY.keys())}")

    camera_class = _CAMERA_REGISTRY[camera_type]
    return camera_class(camera_type)


def _ensure_registered():
    """Ensure default cameras are registered."""
    if _CAMERA_REGISTRY:
        return

    # Import and register default implementations
    try:
        from .oakd import OakDCamera
        register_camera(CameraType.OAK_D, OakDCamera)
        register_camera(CameraType.OAK_D_PRO, OakDCamera)
    except ImportError as e:
        print(f"[CameraFactory] OAK-D not available: {e}")

    try:
        from .zed import ZedCamera
        register_camera(CameraType.ZED, ZedCamera)
        register_camera(CameraType.ZED_2, ZedCamera)
        register_camera(CameraType.ZED_X, ZedCamera)
    except ImportError as e:
        print(f"[CameraFactory] ZED not available: {e}")


def list_available_cameras() -> list:
    """List all registered camera types.

    Returns:
        List of available CameraType values
    """
    _ensure_registered()
    return list(_CAMERA_REGISTRY.keys())
