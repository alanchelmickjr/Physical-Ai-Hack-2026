"""Base Camera Interface

Abstract interface for spatial (depth) cameras.
Implement this for each camera platform.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable, Tuple
from enum import Enum
import numpy as np
import math


class CameraType(Enum):
    """Supported camera types."""
    OAK_D = "oak_d"
    OAK_D_PRO = "oak_d_pro"
    ZED = "zed"
    ZED_X = "zed_x"
    ZED_2 = "zed_2"
    REALSENSE = "realsense"
    REALSENSE_D435 = "realsense_d435"
    REALSENSE_D455 = "realsense_d455"


@dataclass
class SpatialDetection:
    """A detection with 3D spatial coordinates."""
    label: str
    confidence: float

    # 2D bounding box (pixels)
    x1: int
    y1: int
    x2: int
    y2: int

    # 3D position (millimeters, camera frame)
    x: float = 0.0  # Left/right
    y: float = 0.0  # Up/down
    z: float = 0.0  # Depth (distance from camera)

    # Derived
    angle: float = 0.0      # Horizontal angle from camera center (degrees)
    distance: float = 0.0   # Euclidean distance (mm)

    # Tracking
    track_id: int = -1
    timestamp: float = 0.0

    def __post_init__(self):
        """Calculate derived values."""
        if self.z > 0:
            self.angle = math.degrees(math.atan2(self.x, self.z))
            self.distance = math.sqrt(self.x**2 + self.y**2 + self.z**2)


class SpatialCamera(ABC):
    """Abstract interface for depth cameras.

    Implement this for each camera platform to enable
    the same tracking code to work across different hardware.

    Example:
        camera = ZedCamera()
        camera.start()

        detections = camera.get_detections()
        for det in detections:
            print(f"{det.label} at {det.distance}mm")
    """

    def __init__(self):
        self._running = False
        self._on_detection: Optional[Callable[[List[SpatialDetection]], None]] = None
        self._on_frame: Optional[Callable[[np.ndarray], None]] = None

    @property
    @abstractmethod
    def camera_type(self) -> CameraType:
        """Return the camera type."""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Return image width in pixels."""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Return image height in pixels."""
        pass

    @property
    @abstractmethod
    def hfov(self) -> float:
        """Return horizontal field of view in degrees."""
        pass

    @abstractmethod
    def start(self) -> bool:
        """Start the camera.

        Returns:
            True if started successfully
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the camera."""
        pass

    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get current RGB and depth frames.

        Returns:
            (rgb_frame, depth_frame) - either may be None
        """
        pass

    @abstractmethod
    def get_detections(self) -> List[SpatialDetection]:
        """Get current person detections with spatial coordinates.

        Returns:
            List of SpatialDetection objects
        """
        pass

    def set_detection_callback(self, callback: Callable[[List[SpatialDetection]], None]):
        """Set callback for when detections are updated."""
        self._on_detection = callback

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for each RGB frame."""
        self._on_frame = callback

    def get_person_at_angle(
        self, target_angle: float, tolerance: float = 15.0
    ) -> Optional[SpatialDetection]:
        """Find a person at a given angle (for DOA fusion).

        Args:
            target_angle: Target angle in degrees (0 = center, positive = right)
            tolerance: Angle tolerance in degrees

        Returns:
            Best matching detection, or None
        """
        candidates = [
            d for d in self.get_detections()
            if d.label == 'person' and abs(d.angle - target_angle) < tolerance
        ]

        if not candidates:
            return None

        return min(candidates, key=lambda d: abs(d.angle - target_angle))

    def get_closest_person(self) -> Optional[SpatialDetection]:
        """Get the closest detected person."""
        people = [d for d in self.get_detections() if d.label == 'person' and d.z > 0]

        if not people:
            return None

        return min(people, key=lambda d: d.distance)

    @property
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running

    def pixel_to_3d(self, px: int, py: int, depth_mm: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to 3D using pinhole model.

        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate
            depth_mm: Depth in millimeters

        Returns:
            (x, y, z) in millimeters
        """
        cx = self.width / 2
        cy = self.height / 2
        fx = self.width / (2 * math.tan(math.radians(self.hfov / 2)))
        fy = fx  # Assume square pixels

        x = (px - cx) * depth_mm / fx
        y = (py - cy) * depth_mm / fy
        z = depth_mm

        return (x, y, z)
