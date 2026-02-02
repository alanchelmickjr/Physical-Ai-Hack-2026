"""Intel RealSense Camera Implementation

Spatial camera using Intel RealSense D400 series.
Supports D435, D435i, D455.

Requires: pip install pyrealsense2
"""

import threading
import time
from typing import Optional, List, Tuple
import numpy as np
import math

from .base import SpatialCamera, SpatialDetection, CameraType

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("WARNING: pyrealsense2 not installed. Run: pip install pyrealsense2")

# For YOLO on Jetson
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class RealSenseCamera(SpatialCamera):
    """Intel RealSense D400 series camera.

    Provides RGB + depth for spatial tracking.
    YOLO runs on Jetson GPU for person detection.
    """

    def __init__(self, camera_type: CameraType = CameraType.REALSENSE_D435):
        """Initialize RealSense camera.

        Args:
            camera_type: REALSENSE_D435 or REALSENSE_D455
        """
        super().__init__()
        self._type = camera_type

        # RealSense objects
        self._pipeline: Optional[rs.pipeline] = None
        self._config: Optional[rs.config] = None
        self._align: Optional[rs.align] = None
        self._profile = None

        # Frames
        self._rgb_frame: Optional[np.ndarray] = None
        self._depth_frame: Optional[np.ndarray] = None
        self._detections: List[SpatialDetection] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # YOLO
        self._yolo = None
        self._yolo_interval = 3
        self._frame_count = 0

        # Camera specs
        if camera_type == CameraType.REALSENSE_D455:
            self._width = 1280
            self._height = 720
            self._hfov = 90.0
        else:  # D435
            self._width = 1280
            self._height = 720
            self._hfov = 87.0

        # Depth intrinsics (filled on start)
        self._depth_intrinsics = None

    @property
    def camera_type(self) -> CameraType:
        return self._type

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def hfov(self) -> float:
        return self._hfov

    def start(self) -> bool:
        """Start the RealSense camera."""
        if not REALSENSE_AVAILABLE:
            print("[RealSense] pyrealsense2 not available")
            return False

        try:
            self._pipeline = rs.pipeline()
            self._config = rs.config()

            # Configure streams
            self._config.enable_stream(
                rs.stream.color,
                self._width, self._height,
                rs.format.bgr8, 30
            )
            self._config.enable_stream(
                rs.stream.depth,
                self._width, self._height,
                rs.format.z16, 30
            )

            # Start pipeline
            self._profile = self._pipeline.start(self._config)

            # Get depth intrinsics for 3D projection
            depth_stream = self._profile.get_stream(rs.stream.depth)
            self._depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

            # Align depth to color
            self._align = rs.align(rs.stream.color)

            # Load YOLO
            if YOLO_AVAILABLE:
                self._yolo = YOLO("yolov8n.pt")
                print("[RealSense] YOLO loaded for person detection")

            self._running = True
            self._thread = threading.Thread(target=self._process_loop, daemon=True)
            self._thread.start()

            print(f"[RealSense] Started {self._type.value} at {self._width}x{self._height}")
            return True

        except Exception as e:
            print(f"[RealSense] Failed to start: {e}")
            return False

    def stop(self):
        """Stop the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        print("[RealSense] Stopped")

    def _process_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                # Wait for frames
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)

                # Align depth to color
                aligned_frames = self._align.process(frames)

                # Get frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Convert to numpy
                self._rgb_frame = np.asanyarray(color_frame.get_data())
                self._depth_frame = np.asanyarray(depth_frame.get_data())

                # Frame callback
                if self._on_frame and self._rgb_frame is not None:
                    self._on_frame(self._rgb_frame)

                # Run YOLO periodically
                self._frame_count += 1
                if self._yolo and self._frame_count % self._yolo_interval == 0:
                    self._run_detection()

            except Exception as e:
                print(f"[RealSense] Error: {e}")

            time.sleep(0.001)

    def _run_detection(self):
        """Run YOLO detection and get spatial coordinates."""
        if self._rgb_frame is None or self._yolo is None:
            return

        detections = []

        # Run YOLO
        results = self._yolo(self._rgb_frame, verbose=False, classes=[0])  # class 0 = person

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                if conf < 0.5:
                    continue

                # Get center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Get 3D coordinates
                x, y, z = self._get_3d_coords(cx, cy)

                det = SpatialDetection(
                    label="person",
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    x=x, y=y, z=z,
                    timestamp=time.time()
                )
                detections.append(det)

        with self._lock:
            self._detections = detections

        if self._on_detection and detections:
            self._on_detection(detections)

    def _get_3d_coords(self, px: int, py: int) -> Tuple[float, float, float]:
        """Get 3D coordinates from depth.

        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate

        Returns:
            (x, y, z) in millimeters
        """
        if self._depth_frame is None:
            return (0.0, 0.0, 0.0)

        try:
            # Get depth value (in millimeters for RealSense)
            depth = self._depth_frame[py, px]

            if depth <= 0:
                return (0.0, 0.0, 0.0)

            # Use RealSense intrinsics for deprojection
            if self._depth_intrinsics:
                point = rs.rs2_deproject_pixel_to_point(
                    self._depth_intrinsics,
                    [px, py],
                    depth
                )
                return (float(point[0]), float(point[1]), float(point[2]))

            # Fallback to pinhole model
            return self.pixel_to_3d(px, py, depth)

        except Exception:
            return (0.0, 0.0, 0.0)

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get current RGB and depth frames."""
        return (self._rgb_frame, self._depth_frame)

    def get_detections(self) -> List[SpatialDetection]:
        """Get current detections."""
        with self._lock:
            return self._detections.copy()

    def get_depth_at_pixel(self, px: int, py: int) -> float:
        """Get depth at a specific pixel.

        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate

        Returns:
            Depth in millimeters
        """
        if self._depth_frame is None:
            return 0.0

        try:
            return float(self._depth_frame[py, px])
        except:
            return 0.0
