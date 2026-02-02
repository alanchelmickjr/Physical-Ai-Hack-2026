#!/usr/bin/env python3
"""Visual Safety System - Fire & Hazard Detection

Uses OAK-D camera to detect:
- Fire (flames, fire-colored regions)
- Smoke (gray/white moving regions)
- Person down (fallen person detection)
- Obstacles in path

When detected, triggers immediate alert behavior:
- Point at hazard
- Speak warning
- Log incident

This runs continuously in background when enabled.
"""

import asyncio
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
from enum import Enum, auto

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("[Safety] OpenCV not available - visual detection disabled")


class HazardType(Enum):
    """Types of hazards we can detect."""
    FIRE = auto()
    SMOKE = auto()
    PERSON_DOWN = auto()
    OBSTACLE = auto()
    UNKNOWN = auto()


@dataclass
class HazardDetection:
    """A detected hazard."""
    hazard_type: HazardType
    confidence: float  # 0-1
    x: int  # Pixel x
    y: int  # Pixel y
    width: int
    height: int
    direction_degrees: float  # Estimated DOA from camera center
    timestamp: float = field(default_factory=time.time)


@dataclass
class SafetyConfig:
    """Configuration for visual safety system."""
    enabled: bool = True
    fire_detection: bool = True
    smoke_detection: bool = True
    person_down_detection: bool = False  # Requires YOLO
    obstacle_detection: bool = False

    # Thresholds
    fire_confidence_threshold: float = 0.6
    smoke_confidence_threshold: float = 0.5

    # Camera field of view (for DOA estimation)
    camera_fov_degrees: float = 69.0  # OAK-D horizontal FOV

    # Sensitivity (affects detection thresholds)
    sensitivity: str = "medium"  # low, medium, high

    # Auto-alert behavior
    auto_alert: bool = True
    alert_cooldown_seconds: float = 5.0  # Don't spam alerts


class VisualSafety:
    """Visual hazard detection system.

    Uses simple color-based detection for fire/smoke
    (works without ML models). Can be upgraded to use
    YOLO or dedicated fire detection models.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()

        # Callback when hazard detected
        self._on_hazard: Optional[Callable[[HazardDetection], None]] = None

        # State
        self._running = False
        self._last_alert_time: float = 0.0
        self._frame_count: int = 0

        # Fire detection parameters (HSV ranges for flames)
        # Flames are typically orange-red-yellow
        self._fire_hsv_ranges = [
            # Orange-red flames
            ((0, 100, 200), (20, 255, 255)),
            # Yellow flames
            ((20, 100, 200), (35, 255, 255)),
            # Bright red
            ((160, 100, 200), (180, 255, 255)),
        ]

        # Smoke detection (gray/white regions with motion)
        self._smoke_hsv_range = ((0, 0, 150), (180, 50, 255))  # Low saturation, high value

        # Previous frame for motion detection
        self._prev_gray: Optional[np.ndarray] = None

    def set_hazard_callback(self, callback: Callable[[HazardDetection], None]):
        """Set callback for when hazard is detected."""
        self._on_hazard = callback

    def set_sensitivity(self, sensitivity: str):
        """Set detection sensitivity."""
        self.config.sensitivity = sensitivity
        # Adjust thresholds
        thresholds = {
            "low": (0.8, 0.7),
            "medium": (0.6, 0.5),
            "high": (0.4, 0.3),
        }
        fire_thresh, smoke_thresh = thresholds.get(sensitivity, (0.6, 0.5))
        self.config.fire_confidence_threshold = fire_thresh
        self.config.smoke_confidence_threshold = smoke_thresh

    def process_frame(self, frame: np.ndarray) -> List[HazardDetection]:
        """Process a single frame for hazards.

        Args:
            frame: BGR image from camera

        Returns:
            List of detected hazards
        """
        if not HAS_OPENCV or not self.config.enabled:
            return []

        detections = []
        height, width = frame.shape[:2]

        # Fire detection
        if self.config.fire_detection:
            fire_det = self._detect_fire(frame, width, height)
            if fire_det:
                detections.append(fire_det)

        # Smoke detection
        if self.config.smoke_detection:
            smoke_det = self._detect_smoke(frame, width, height)
            if smoke_det:
                detections.append(smoke_det)

        # Alert on detections
        if detections and self.config.auto_alert:
            self._handle_detections(detections)

        self._frame_count += 1
        return detections

    def _detect_fire(self, frame: np.ndarray, width: int, height: int) -> Optional[HazardDetection]:
        """Detect fire using color-based method."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Combine masks for different flame colors
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for low, high in self._fire_hsv_ranges:
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest fire-colored region
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Minimum area threshold (adjusts with sensitivity)
        min_area = {"low": 5000, "medium": 2000, "high": 500}.get(self.config.sensitivity, 2000)

        if area < min_area:
            return None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)
        center_x = x + w // 2

        # Calculate confidence based on area and color intensity
        confidence = min(1.0, area / 10000)

        # Check for flickering (fire characteristic) using motion
        if self._prev_gray is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_prev = self._prev_gray[y:y+h, x:x+w]
            roi_curr = gray[y:y+h, x:x+w]
            if roi_prev.shape == roi_curr.shape:
                diff = cv2.absdiff(roi_prev, roi_curr)
                motion = np.mean(diff)
                # Fire flickers - boost confidence if there's motion
                if motion > 10:
                    confidence = min(1.0, confidence * 1.3)

        if confidence < self.config.fire_confidence_threshold:
            return None

        # Convert pixel position to direction
        direction = self._pixel_to_direction(center_x, width)

        return HazardDetection(
            hazard_type=HazardType.FIRE,
            confidence=confidence,
            x=x, y=y, width=w, height=h,
            direction_degrees=direction
        )

    def _detect_smoke(self, frame: np.ndarray, width: int, height: int) -> Optional[HazardDetection]:
        """Detect smoke using color + motion."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        # Motion detection
        diff = cv2.absdiff(self._prev_gray, gray)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Color detection (gray/white regions)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low, high = self._smoke_hsv_range
        color_mask = cv2.inRange(hsv, np.array(low), np.array(high))

        # Combine: smoke is gray/white AND moving
        smoke_mask = cv2.bitwise_and(motion_mask, color_mask)

        # Clean up
        kernel = np.ones((7, 7), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)

        # Update previous frame
        self._prev_gray = gray

        # Find contours
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest smoke region
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        min_area = {"low": 8000, "medium": 4000, "high": 1500}.get(self.config.sensitivity, 4000)

        if area < min_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        center_x = x + w // 2

        confidence = min(1.0, area / 15000)

        if confidence < self.config.smoke_confidence_threshold:
            return None

        direction = self._pixel_to_direction(center_x, width)

        return HazardDetection(
            hazard_type=HazardType.SMOKE,
            confidence=confidence,
            x=x, y=y, width=w, height=h,
            direction_degrees=direction
        )

    def _pixel_to_direction(self, pixel_x: int, frame_width: int) -> float:
        """Convert pixel x position to direction in degrees.

        Center = 0°, left = negative, right = positive
        """
        # Normalize to -1 to 1
        normalized = (pixel_x - frame_width / 2) / (frame_width / 2)
        # Scale by half FOV
        direction = normalized * (self.config.camera_fov_degrees / 2)
        return direction

    def _handle_detections(self, detections: List[HazardDetection]):
        """Handle detected hazards."""
        now = time.time()

        # Cooldown to prevent alert spam
        if now - self._last_alert_time < self.config.alert_cooldown_seconds:
            return

        # Find highest priority/confidence detection
        # Fire > Smoke > Others
        priority = {HazardType.FIRE: 0, HazardType.SMOKE: 1, HazardType.PERSON_DOWN: 2}
        detections.sort(key=lambda d: (priority.get(d.hazard_type, 99), -d.confidence))

        detection = detections[0]

        self._last_alert_time = now

        # Call callback
        if self._on_hazard:
            self._on_hazard(detection)

        # Log
        print(f"[SAFETY] {detection.hazard_type.name} detected! "
              f"Confidence: {detection.confidence:.1%}, Direction: {detection.direction_degrees:.0f}°")


# Singleton
_safety: Optional[VisualSafety] = None


def get_visual_safety() -> VisualSafety:
    """Get the singleton visual safety system."""
    global _safety
    if _safety is None:
        _safety = VisualSafety()
    return _safety


# Test
if __name__ == "__main__":
    print("Visual Safety System")
    print("=" * 50)

    if not HAS_OPENCV:
        print("OpenCV required for testing")
        exit(1)

    safety = get_visual_safety()

    def on_hazard(detection: HazardDetection):
        print(f"\n🔥 ALERT: {detection.hazard_type.name}")
        print(f"   Confidence: {detection.confidence:.1%}")
        print(f"   Direction: {detection.direction_degrees:.0f}°")
        print(f"   Location: ({detection.x}, {detection.y})")

    safety.set_hazard_callback(on_hazard)

    # Test with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available")
        exit(1)

    print("\nRunning fire/smoke detection on camera...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = safety.process_frame(frame)

        # Draw detections
        for det in detections:
            color = (0, 0, 255) if det.hazard_type == HazardType.FIRE else (128, 128, 128)
            cv2.rectangle(frame, (det.x, det.y), (det.x + det.width, det.y + det.height), color, 2)
            cv2.putText(frame, f"{det.hazard_type.name} {det.confidence:.0%}",
                       (det.x, det.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Safety Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
