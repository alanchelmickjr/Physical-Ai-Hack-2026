"""ZED Camera Implementation

Spatial camera using Stereolabs ZED cameras.
Supports ZED, ZED 2, ZED X, ZED Mini.

Requires: pip install pyzed (ZED SDK must be installed first)
"""

import threading
import time
from typing import Optional, List, Tuple
import numpy as np
import math

from .base import SpatialCamera, SpatialDetection, CameraType

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("WARNING: pyzed not installed. Install ZED SDK first, then: pip install pyzed")

# For YOLO on Jetson (when ZED's built-in detection isn't used)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ZedCamera(SpatialCamera):
    """ZED spatial camera with stereo depth.

    The ZED SDK provides:
    - High-quality stereo depth
    - Built-in object detection (optional)
    - Spatial mapping / SLAM
    - Body tracking

    For Booster K1, we use ZED X with GMSL2 connection.
    """

    def __init__(self, camera_type: CameraType = CameraType.ZED_X):
        """Initialize ZED camera.

        Args:
            camera_type: ZED variant (ZED, ZED_2, ZED_X)
        """
        super().__init__()
        self._type = camera_type

        # ZED objects
        self._zed: Optional[sl.Camera] = None
        self._runtime_params: Optional[sl.RuntimeParameters] = None

        # Frames
        self._rgb_mat = None
        self._depth_mat = None
        self._point_cloud = None

        # State
        self._detections: List[SpatialDetection] = []
        self._rgb_frame: Optional[np.ndarray] = None
        self._depth_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # YOLO for person detection (runs on Jetson GPU)
        self._yolo = None
        self._yolo_interval = 3  # Run YOLO every N frames
        self._frame_count = 0

        # Camera specs (ZED X defaults)
        self._width = 1920
        self._height = 1080
        self._hfov = 110.0  # ZED X wide FOV

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
        """Start the ZED camera."""
        if not ZED_AVAILABLE:
            print("[ZedCamera] pyzed not available. Install ZED SDK first.")
            return False

        try:
            self._zed = sl.Camera()

            # Init params
            init_params = sl.InitParameters()

            # Set camera model specific params
            if self._type == CameraType.ZED_X:
                init_params.camera_resolution = sl.RESOLUTION.HD1080
                init_params.camera_fps = 30
                self._width = 1920
                self._height = 1080
                self._hfov = 110.0
            elif self._type == CameraType.ZED_2:
                init_params.camera_resolution = sl.RESOLUTION.HD1080
                init_params.camera_fps = 30
                self._width = 1920
                self._height = 1080
                self._hfov = 110.0
            else:  # ZED original
                init_params.camera_resolution = sl.RESOLUTION.HD720
                init_params.camera_fps = 30
                self._width = 1280
                self._height = 720
                self._hfov = 90.0

            init_params.depth_mode = sl.DEPTH_MODE.ULTRA
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.depth_minimum_distance = 200  # 20cm
            init_params.depth_maximum_distance = 20000  # 20m

            # Open camera
            status = self._zed.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"[ZedCamera] Failed to open: {status}")
                return False

            # Runtime params
            self._runtime_params = sl.RuntimeParameters()
            self._runtime_params.confidence_threshold = 50
            self._runtime_params.texture_confidence_threshold = 100

            # Allocate matrices
            self._rgb_mat = sl.Mat()
            self._depth_mat = sl.Mat()
            self._point_cloud = sl.Mat()

            # Load YOLO for person detection
            if YOLO_AVAILABLE:
                self._yolo = YOLO("yolov8n.pt")
                print("[ZedCamera] YOLO loaded for person detection")

            self._running = True
            self._thread = threading.Thread(target=self._process_loop, daemon=True)
            self._thread.start()

            print(f"[ZedCamera] Started ({self._type.value}) at {self._width}x{self._height}")
            return True

        except Exception as e:
            print(f"[ZedCamera] Failed to start: {e}")
            return False

    def stop(self):
        """Stop the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._zed:
            self._zed.close()
            self._zed = None
        print("[ZedCamera] Stopped")

    def _process_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                if self._zed.grab(self._runtime_params) == sl.ERROR_CODE.SUCCESS:
                    # Get RGB image
                    self._zed.retrieve_image(self._rgb_mat, sl.VIEW.LEFT)
                    self._rgb_frame = self._rgb_mat.get_data()[:, :, :3].copy()

                    # Get depth map
                    self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
                    self._depth_frame = self._depth_mat.get_data().copy()

                    # Get point cloud for 3D coords
                    self._zed.retrieve_measure(self._point_cloud, sl.MEASURE.XYZRGBA)

                    # Frame callback
                    if self._on_frame and self._rgb_frame is not None:
                        self._on_frame(self._rgb_frame)

                    # Run YOLO periodically
                    self._frame_count += 1
                    if self._yolo and self._frame_count % self._yolo_interval == 0:
                        self._run_detection()

            except Exception as e:
                print(f"[ZedCamera] Error: {e}")

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
                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                if conf < 0.5:
                    continue

                # Get center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Get 3D coordinates from point cloud
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
        """Get 3D coordinates from point cloud.

        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate

        Returns:
            (x, y, z) in millimeters
        """
        if self._point_cloud is None:
            return (0.0, 0.0, 0.0)

        try:
            # Get point from point cloud
            err, point = self._point_cloud.get_value(px, py)

            if err == sl.ERROR_CODE.SUCCESS and not math.isnan(point[2]):
                return (float(point[0]), float(point[1]), float(point[2]))

            # Fallback: use depth map
            if self._depth_frame is not None:
                depth = self._depth_frame[py, px]
                if not math.isnan(depth) and depth > 0:
                    return self.pixel_to_3d(px, py, depth)

        except Exception:
            pass

        return (0.0, 0.0, 0.0)

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get current RGB and depth frames."""
        return (self._rgb_frame, self._depth_frame)

    def get_detections(self) -> List[SpatialDetection]:
        """Get current detections."""
        with self._lock:
            return self._detections.copy()

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Get current point cloud data.

        Returns:
            Point cloud as numpy array (H, W, 4) with XYZRGBA
        """
        if self._point_cloud:
            return self._point_cloud.get_data()
        return None

    def enable_positional_tracking(self) -> bool:
        """Enable ZED's built-in positional tracking (SLAM).

        Returns:
            True if enabled successfully
        """
        if not self._zed:
            return False

        tracking_params = sl.PositionalTrackingParameters()
        status = self._zed.enable_positional_tracking(tracking_params)
        return status == sl.ERROR_CODE.SUCCESS

    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """Get camera position from positional tracking.

        Returns:
            (x, y, z) position in meters, or None
        """
        if not self._zed:
            return None

        pose = sl.Pose()
        state = self._zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)

        if state == sl.POSITIONAL_TRACKING_STATE.OK:
            translation = pose.get_translation(sl.Translation())
            return (translation.get()[0], translation.get()[1], translation.get()[2])

        return None
