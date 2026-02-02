"""OAK-D Camera Implementation

Spatial camera using Luxonis OAK-D with onboard VPU for YOLO.
Supports OAK-D, OAK-D Pro, OAK-D Lite.
"""

import threading
import time
from typing import Optional, List, Tuple
from pathlib import Path
import urllib.request
import numpy as np

from .base import SpatialCamera, SpatialDetection, CameraType

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False
    print("WARNING: depthai not installed. Run: pip install depthai --break-system-packages")


# YOLO model for VPU
YOLO_BLOB_URL = "https://raw.githubusercontent.com/luxonis/depthai-model-zoo/main/models/yolov8n_coco_640x352/yolov8n_coco_640x352_6shave.blob"
YOLO_BLOB_PATH = Path.home() / ".cache" / "depthai" / "yolov8n_coco_640x352_6shave.blob"

YOLO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class OakDCamera(SpatialCamera):
    """OAK-D spatial camera with VPU-accelerated YOLO.

    Uses the Myriad X VPU for onboard YOLO inference,
    freeing the Jetson GPU for other tasks.
    """

    # OAK-D Pro specs
    RGB_WIDTH = 640
    RGB_HEIGHT = 352  # YOLO input size
    HFOV = 69.0

    def __init__(self, variant: str = "pro"):
        """Initialize OAK-D camera.

        Args:
            variant: "oak_d", "oak_d_pro", or "oak_d_lite"
        """
        super().__init__()
        self._variant = variant
        self._type = CameraType.OAK_D_PRO if variant == "pro" else CameraType.OAK_D

        self.pipeline: Optional[dai.Pipeline] = None
        self.device: Optional[dai.Device] = None

        self._rgb_queue = None
        self._depth_queue = None
        self._detection_queue = None

        self._detections: List[SpatialDetection] = []
        self._depth_frame: Optional[np.ndarray] = None
        self._rgb_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def camera_type(self) -> CameraType:
        return self._type

    @property
    def width(self) -> int:
        return self.RGB_WIDTH

    @property
    def height(self) -> int:
        return self.RGB_HEIGHT

    @property
    def hfov(self) -> float:
        return self.HFOV

    def _download_blob(self) -> str:
        """Download YOLO blob if not present."""
        if YOLO_BLOB_PATH.exists():
            return str(YOLO_BLOB_PATH)

        print(f"[OakDCamera] Downloading YOLO blob for VPU...")
        YOLO_BLOB_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YOLO_BLOB_URL, YOLO_BLOB_PATH)
        print(f"[OakDCamera] Downloaded to {YOLO_BLOB_PATH}")
        return str(YOLO_BLOB_PATH)

    def _create_pipeline(self) -> dai.Pipeline:
        """Create OAK-D pipeline with YOLO on VPU + stereo depth."""
        pipeline = dai.Pipeline()

        # RGB Camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 352)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        # Mono cameras for stereo depth
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        # Stereo depth
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        # YOLO Spatial Detection on VPU
        blob_path = self._download_blob()
        spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        spatial_nn.setBlobPath(blob_path)
        spatial_nn.setConfidenceThreshold(0.5)
        spatial_nn.input.setBlocking(False)
        spatial_nn.setBoundingBoxScaleFactor(0.5)
        spatial_nn.setDepthLowerThreshold(100)
        spatial_nn.setDepthUpperThreshold(10000)

        # YOLOv8 settings
        spatial_nn.setNumClasses(80)
        spatial_nn.setCoordinateSize(4)
        spatial_nn.setAnchors([])
        spatial_nn.setAnchorMasks({})
        spatial_nn.setIouThreshold(0.5)

        # Linking
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        cam_rgb.preview.link(spatial_nn.input)
        stereo.depth.link(spatial_nn.inputDepth)

        # Outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("detections")
        spatial_nn.out.link(xout_nn.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        return pipeline

    def start(self) -> bool:
        """Start the OAK-D camera."""
        if not DEPTHAI_AVAILABLE:
            print("[OakDCamera] depthai not available")
            return False

        try:
            self.pipeline = self._create_pipeline()
            self.device = dai.Device(self.pipeline)

            self._rgb_queue = self.device.getOutputQueue(
                name="rgb", maxSize=4, blocking=False
            )
            self._depth_queue = self.device.getOutputQueue(
                name="depth", maxSize=4, blocking=False
            )
            self._detection_queue = self.device.getOutputQueue(
                name="detections", maxSize=4, blocking=False
            )

            self._running = True
            self._thread = threading.Thread(target=self._process_loop, daemon=True)
            self._thread.start()

            print("[OakDCamera] Started with YOLO on VPU + stereo depth")
            return True

        except Exception as e:
            print(f"[OakDCamera] Failed to start: {e}")
            return False

    def stop(self):
        """Stop the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.device:
            self.device.close()
            self.device = None
        print("[OakDCamera] Stopped")

    def _process_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                rgb_msg = self._rgb_queue.tryGet()
                depth_msg = self._depth_queue.tryGet()
                det_msg = self._detection_queue.tryGet()

                if rgb_msg is not None:
                    self._rgb_frame = rgb_msg.getCvFrame()
                    if self._on_frame:
                        self._on_frame(self._rgb_frame)

                if depth_msg is not None:
                    self._depth_frame = depth_msg.getFrame()

                if det_msg is not None:
                    self._process_detections(det_msg)

            except Exception as e:
                print(f"[OakDCamera] Error: {e}")

            time.sleep(0.001)

    def _process_detections(self, det_msg):
        """Process VPU detections."""
        detections = []

        for detection in det_msg.detections:
            label_idx = detection.label
            label = YOLO_LABELS[label_idx] if label_idx < len(YOLO_LABELS) else f"class_{label_idx}"

            if label != 'person':
                continue

            x1 = int(detection.xmin * 640)
            y1 = int(detection.ymin * 352)
            x2 = int(detection.xmax * 640)
            y2 = int(detection.ymax * 352)

            x = detection.spatialCoordinates.x
            y = detection.spatialCoordinates.y
            z = detection.spatialCoordinates.z

            det = SpatialDetection(
                label=label,
                confidence=detection.confidence,
                x1=x1, y1=y1, x2=x2, y2=y2,
                x=x, y=y, z=z,
                timestamp=time.time()
            )
            detections.append(det)

        with self._lock:
            self._detections = detections

        if self._on_detection and detections:
            self._on_detection(detections)

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get current RGB and depth frames."""
        return (self._rgb_frame, self._depth_frame)

    def get_detections(self) -> List[SpatialDetection]:
        """Get current detections."""
        with self._lock:
            return self._detections.copy()
