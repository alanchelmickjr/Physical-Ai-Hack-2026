#!/usr/bin/env python3
"""Spatial Tracker - OAK-D depth-enabled person tracking

Uses OAK-D Pro stereo depth + RGB to provide 3D positions for detected people.
Enables DOA-to-person fusion by matching audio direction with spatial position.

Usage:
    from spatial_tracker import SpatialTracker, get_spatial_tracker

    tracker = get_spatial_tracker()
    tracker.start()

    for detection in tracker.get_detections():
        print(f"{detection.label} at {detection.x:.0f}mm, {detection.y:.0f}mm, {detection.z:.0f}mm")
        print(f"  Angle: {detection.angle:.1f}°, Distance: {detection.distance:.0f}mm")
"""

import depthai as dai
import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from enum import Enum
import math

# Optional visual safety import
try:
    from visual_safety import VisualSafety, HazardDetection, get_visual_safety
    HAS_VISUAL_SAFETY = True
except ImportError:
    HAS_VISUAL_SAFETY = False


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
            # Angle from camera center (positive = right)
            self.angle = math.degrees(math.atan2(self.x, self.z))
            # Euclidean distance
            self.distance = math.sqrt(self.x**2 + self.y**2 + self.z**2)


class SpatialTracker:
    """OAK-D spatial tracking with stereo depth.

    Provides 3D positions for detected people, enabling:
    - DOA-to-person fusion (match audio direction with visual position)
    - Distance-based attention (closer people get priority)
    - SLAM groundwork (spatial mapping)
    """

    # OAK-D Pro specs
    RGB_WIDTH = 640
    RGB_HEIGHT = 480
    HFOV = 69.0  # Horizontal FOV in degrees

    def __init__(self):
        self.pipeline: Optional[dai.Pipeline] = None
        self.device: Optional[dai.Device] = None

        # Queues
        self._rgb_queue = None
        self._depth_queue = None
        self._detection_queue = None

        # State
        self._running = False
        self._detections: List[SpatialDetection] = []
        self._depth_frame: Optional[np.ndarray] = None
        self._rgb_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # Callbacks
        self._on_detection: Optional[Callable[[List[SpatialDetection]], None]] = None
        self._on_frame: Optional[Callable[[np.ndarray], None]] = None

        # YOLO (external, run on Jetson)
        self._yolo = None
        self._yolo_interval = 8  # Run YOLO every N frames
        self._frame_count = 0

        # Visual safety integration
        self._visual_safety = None
        self._safety_enabled = False
        self._on_hazard: Optional[Callable] = None

    def set_detection_callback(self, callback: Callable[[List[SpatialDetection]], None]):
        """Set callback for when detections are updated."""
        self._on_detection = callback

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for each RGB frame (for visual safety, etc)."""
        self._on_frame = callback

    def enable_visual_safety(self, enabled: bool = True,
                              hazard_callback: Optional[Callable] = None) -> bool:
        """Enable/disable visual safety detection (fire, smoke, etc).

        Args:
            enabled: Whether to enable safety detection
            hazard_callback: Optional callback for hazard events

        Returns:
            True if safety system was enabled successfully
        """
        if not HAS_VISUAL_SAFETY:
            print("[SpatialTracker] Visual safety module not available")
            return False

        self._safety_enabled = enabled
        if enabled:
            self._visual_safety = get_visual_safety()
            if hazard_callback:
                self._on_hazard = hazard_callback
                self._visual_safety.set_hazard_callback(hazard_callback)
            print("[SpatialTracker] Visual safety enabled")
        else:
            self._visual_safety = None
            print("[SpatialTracker] Visual safety disabled")
        return True

    def set_hazard_callback(self, callback: Callable):
        """Set callback for hazard detection (fire, smoke, etc)."""
        self._on_hazard = callback
        if self._visual_safety:
            self._visual_safety.set_hazard_callback(callback)

    def _create_pipeline(self) -> dai.Pipeline:
        """Create OAK-D pipeline with RGB + stereo depth."""
        pipeline = dai.Pipeline()

        # RGB Camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(self.RGB_WIDTH, self.RGB_HEIGHT)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        # Mono cameras for stereo depth
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Left
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Right

        # Stereo depth
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to RGB
        stereo.setOutputSize(self.RGB_WIDTH, self.RGB_HEIGHT)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

        # Linking
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Output queues
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        return pipeline

    def start(self) -> bool:
        """Start the spatial tracker."""
        try:
            self.pipeline = self._create_pipeline()
            self.device = dai.Device(self.pipeline)

            self._rgb_queue = self.device.getOutputQueue(
                name="rgb", maxSize=4, blocking=False
            )
            self._depth_queue = self.device.getOutputQueue(
                name="depth", maxSize=4, blocking=False
            )

            self._running = True

            # Load YOLO for person detection
            try:
                from ultralytics import YOLO
                self._yolo = YOLO('yolov8n.pt')
            except Exception as e:
                print(f"[SpatialTracker] YOLO not available: {e}")

            # Start processing thread
            self._thread = threading.Thread(target=self._process_loop, daemon=True)
            self._thread.start()

            print("[SpatialTracker] Started with stereo depth")
            return True

        except Exception as e:
            print(f"[SpatialTracker] Failed to start: {e}")
            return False

    def stop(self):
        """Stop the spatial tracker."""
        self._running = False
        if self.device:
            self.device.close()
            self.device = None

    def _process_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                # Get frames
                rgb_msg = self._rgb_queue.tryGet()
                depth_msg = self._depth_queue.tryGet()

                if rgb_msg is not None:
                    self._rgb_frame = rgb_msg.getCvFrame()
                    self._frame_count += 1

                    # Visual safety check (fire, smoke detection)
                    if self._safety_enabled and self._visual_safety and self._rgb_frame is not None:
                        self._visual_safety.process_frame(self._rgb_frame)

                    # Call frame callback (for custom processing)
                    if self._on_frame and self._rgb_frame is not None:
                        self._on_frame(self._rgb_frame)

                if depth_msg is not None:
                    self._depth_frame = depth_msg.getFrame()

                # Run detection periodically
                if self._frame_count % self._yolo_interval == 0:
                    self._run_detection()

            except Exception as e:
                print(f"[SpatialTracker] Error: {e}")

            time.sleep(0.001)  # Prevent busy loop

    def _run_detection(self):
        """Run YOLO detection and add depth."""
        if self._rgb_frame is None or self._yolo is None:
            return

        detections = []

        try:
            results = self._yolo(self._rgb_frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = result.names[cls]

                    # Only track people for now
                    if label != 'person':
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get depth at center of bbox
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    x, y, z = self._get_spatial_coords(cx, cy)

                    detection = SpatialDetection(
                        label=label,
                        confidence=conf,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        x=x, y=y, z=z,
                        timestamp=time.time()
                    )
                    detections.append(detection)

        except Exception as e:
            print(f"[SpatialTracker] Detection error: {e}")

        with self._lock:
            self._detections = detections

        if self._on_detection and detections:
            self._on_detection(detections)

    def _get_spatial_coords(self, px: int, py: int) -> tuple:
        """Get 3D coordinates for a pixel location.

        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate

        Returns:
            (x, y, z) in millimeters, camera frame
        """
        if self._depth_frame is None:
            return (0.0, 0.0, 0.0)

        # Get depth value (already in mm for OAK-D)
        # Average over small region for stability
        region_size = 10
        x1 = max(0, px - region_size)
        x2 = min(self._depth_frame.shape[1], px + region_size)
        y1 = max(0, py - region_size)
        y2 = min(self._depth_frame.shape[0], py + region_size)

        region = self._depth_frame[y1:y2, x1:x2]
        valid = region[region > 0]

        if len(valid) == 0:
            return (0.0, 0.0, 0.0)

        z = float(np.median(valid))  # Depth in mm

        # Convert pixel to 3D using pinhole camera model
        # x = (px - cx) * z / fx
        cx = self.RGB_WIDTH / 2
        cy = self.RGB_HEIGHT / 2
        fx = self.RGB_WIDTH / (2 * math.tan(math.radians(self.HFOV / 2)))
        fy = fx  # Assume square pixels

        x = (px - cx) * z / fx
        y = (py - cy) * z / fy

        return (x, y, z)

    def get_detections(self) -> List[SpatialDetection]:
        """Get current detections with spatial coordinates."""
        with self._lock:
            return self._detections.copy()

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Get current RGB frame."""
        return self._rgb_frame

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Get current depth frame."""
        return self._depth_frame

    def get_person_at_angle(self, target_angle: float, tolerance: float = 15.0) -> Optional[SpatialDetection]:
        """Find a person at a given angle (for DOA fusion).

        Args:
            target_angle: Target angle in degrees (0 = center, positive = right)
            tolerance: Angle tolerance in degrees

        Returns:
            Best matching detection, or None
        """
        with self._lock:
            candidates = [
                d for d in self._detections
                if d.label == 'person' and abs(d.angle - target_angle) < tolerance
            ]

        if not candidates:
            return None

        # Return closest match by angle
        return min(candidates, key=lambda d: abs(d.angle - target_angle))

    def get_closest_person(self) -> Optional[SpatialDetection]:
        """Get the closest detected person."""
        with self._lock:
            people = [d for d in self._detections if d.label == 'person' and d.z > 0]

        if not people:
            return None

        return min(people, key=lambda d: d.distance)


# =============================================================================
# Singleton
# =============================================================================
_tracker: Optional[SpatialTracker] = None


def get_spatial_tracker() -> SpatialTracker:
    """Get the singleton spatial tracker."""
    global _tracker
    if _tracker is None:
        _tracker = SpatialTracker()
    return _tracker


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    import cv2

    tracker = get_spatial_tracker()

    def on_detection(detections):
        for d in detections:
            print(f"[{d.label}] conf={d.confidence:.2f} "
                  f"pos=({d.x:.0f}, {d.y:.0f}, {d.z:.0f})mm "
                  f"angle={d.angle:.1f}° dist={d.distance:.0f}mm")

    tracker.set_detection_callback(on_detection)
    tracker.start()

    print("Spatial tracker running. Press Ctrl+C to exit.")
    try:
        while True:
            frame = tracker.get_rgb_frame()
            depth = tracker.get_depth_frame()

            if frame is not None:
                # Draw detections
                for d in tracker.get_detections():
                    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                    label = f"{d.label} {d.distance/1000:.1f}m {d.angle:.0f}°"
                    cv2.putText(frame, label, (d.x1, d.y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Spatial Tracker", frame)

            if depth is not None:
                # Normalize depth for display
                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("Depth", depth_vis)

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
