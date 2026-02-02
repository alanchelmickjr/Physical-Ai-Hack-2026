"""Camera Abstraction Layer

Provides a unified interface for different depth cameras.
Drop in a new camera by implementing SpatialCamera.

Usage:
    from cameras import get_camera, CameraType

    camera = get_camera(CameraType.ZED_X)
    camera.start()

    frame, depth, detections = camera.get_frame()
"""

from .base import SpatialCamera, SpatialDetection, CameraType
from .factory import get_camera, register_camera

__all__ = [
    'SpatialCamera',
    'SpatialDetection',
    'CameraType',
    'get_camera',
    'register_camera',
]
