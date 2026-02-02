#!/usr/bin/env python3
"""DOA + Spatial Fusion - Active speaker identification

Combines ReSpeaker DOA with OAK-D spatial detection to identify
which detected person is currently speaking.

The fusion algorithm:
1. DOA gives direction of audio source (0-360°)
2. Spatial tracker gives 3D positions of detected people
3. Each detection is scored by:
   - Angular match: how close DOA is to detection's angle
   - Depth confidence: closer people have more reliable depth
4. Highest-scoring detection = active speaker

Usage:
    from doa_spatial_fusion import DOASpatialFusion, get_fusion

    fusion = get_fusion()
    fusion.start()

    # Get active speaker (person most likely to be speaking)
    speaker = fusion.get_active_speaker()
    if speaker:
        print(f"Speaker at {speaker.distance:.0f}mm, {speaker.angle:.1f}°")

    # Get speaker identity if known
    identity = fusion.get_speaker_identity()
    if identity:
        print(f"Speaking: {identity['name']} (confidence: {identity['confidence']:.0%})")
"""

import threading
import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
from enum import Enum

# Local imports
from doa_reader import ReSpeakerDOA, get_doa
from spatial_tracker import SpatialTracker, SpatialDetection, get_spatial_tracker


class FusionConfidence(Enum):
    """Confidence levels for speaker identification."""
    HIGH = "high"       # DOA + visual + depth all agree
    MEDIUM = "medium"   # DOA + visual agree, depth uncertain
    LOW = "low"         # DOA only, no visual match
    NONE = "none"       # No active speaker detected


@dataclass
class SpeakerMatch:
    """Result of DOA-to-spatial fusion."""
    detection: Optional[SpatialDetection] = None
    doa_angle: float = 0.0          # DOA from ReSpeaker
    angle_error: float = 0.0        # Difference between DOA and detection angle
    depth_confidence: float = 0.0   # How reliable is the depth reading
    fusion_score: float = 0.0       # Combined score (0-1)
    confidence: FusionConfidence = FusionConfidence.NONE
    timestamp: float = field(default_factory=time.time)

    @property
    def is_valid(self) -> bool:
        """Check if this is a valid speaker match."""
        return self.detection is not None and self.fusion_score > 0.3


class DOASpatialFusion:
    """Fuses ReSpeaker DOA with OAK-D spatial detection.

    This enables:
    - Active speaker identification in multi-person scenes
    - Depth-confirmed identity (don't confuse people at different distances)
    - Robust tracking even with partial visual occlusion
    """

    # Fusion parameters
    DOA_SIGMA = 15.0       # DOA uncertainty in degrees (±15°)
    MIN_DEPTH = 300.0      # Minimum reliable depth (mm)
    MAX_DEPTH = 5000.0     # Maximum reliable depth (mm)
    ANGLE_WEIGHT = 0.7     # Weight for angular match
    DEPTH_WEIGHT = 0.3     # Weight for depth confidence

    def __init__(self):
        self._doa: Optional[ReSpeakerDOA] = None
        self._tracker: Optional[SpatialTracker] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Current state
        self._current_match: Optional[SpeakerMatch] = None
        self._lock = threading.Lock()

        # Callbacks
        self._on_speaker_change: List[Callable[[SpeakerMatch], None]] = []

        # Identity integration (optional)
        self._identity_callback: Optional[Callable[[SpatialDetection], Optional[Dict]]] = None

        # History for temporal smoothing
        self._match_history: List[SpeakerMatch] = []
        self._history_max = 5

    def set_identity_callback(self, callback: Callable[[SpatialDetection], Optional[Dict]]):
        """Set callback to get identity for a detection.

        The callback receives a SpatialDetection and should return:
        {'name': 'Person Name', 'confidence': 0.85, 'source': 'face'/'voice'}
        or None if identity unknown.
        """
        self._identity_callback = callback

    def on_speaker_change(self, callback: Callable[[SpeakerMatch], None]):
        """Register callback for when active speaker changes."""
        self._on_speaker_change.append(callback)

    def start(self) -> bool:
        """Start the fusion system."""
        try:
            # Get DOA reader
            self._doa = get_doa()
            self._doa.start(poll_rate_hz=30)

            # Get spatial tracker
            self._tracker = get_spatial_tracker()
            if not self._tracker.start():
                print("[DOASpatialFusion] Warning: Spatial tracker failed to start")
                # Continue anyway - we can still use DOA only

            self._running = True
            self._thread = threading.Thread(target=self._fusion_loop, daemon=True)
            self._thread.start()

            print("[DOASpatialFusion] Started")
            return True

        except Exception as e:
            print(f"[DOASpatialFusion] Failed to start: {e}")
            return False

    def stop(self):
        """Stop the fusion system."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._doa:
            self._doa.stop()
        if self._tracker:
            self._tracker.stop()
        print("[DOASpatialFusion] Stopped")

    def _fusion_loop(self):
        """Main fusion loop - matches DOA to spatial detections."""
        while self._running:
            try:
                # Get DOA state
                doa_angle, is_speaking, speech_prob = self._doa.get_doa()

                if not is_speaking or speech_prob < 0.3:
                    # No one speaking - clear match
                    with self._lock:
                        if self._current_match is not None:
                            self._current_match = None
                            self._notify_speaker_change(SpeakerMatch())
                    time.sleep(0.033)  # ~30Hz
                    continue

                # Get spatial detections
                detections = []
                if self._tracker:
                    detections = self._tracker.get_detections()

                # Find best match
                match = self._find_best_match(doa_angle, detections, speech_prob)

                # Update state
                with self._lock:
                    changed = self._is_match_changed(match)
                    self._current_match = match
                    self._match_history.append(match)
                    if len(self._match_history) > self._history_max:
                        self._match_history.pop(0)

                if changed:
                    self._notify_speaker_change(match)

            except Exception as e:
                print(f"[DOASpatialFusion] Error: {e}")

            time.sleep(0.033)  # ~30Hz

    def _find_best_match(self, doa_angle: float, detections: List[SpatialDetection],
                         speech_prob: float) -> SpeakerMatch:
        """Find the detection that best matches the DOA.

        Scoring formula:
        - angular_score = exp(-(angle_error / DOA_SIGMA)^2 / 2)  # Gaussian
        - depth_score = confidence based on distance (closer = higher)
        - fusion_score = ANGLE_WEIGHT * angular_score + DEPTH_WEIGHT * depth_score
        """
        if not detections:
            # No detections - return DOA-only match
            return SpeakerMatch(
                detection=None,
                doa_angle=doa_angle,
                confidence=FusionConfidence.LOW,
                fusion_score=speech_prob * 0.5  # Reduced confidence without visual
            )

        best_match = None
        best_score = 0.0

        for det in detections:
            if det.label != 'person':
                continue

            # Angular score (Gaussian likelihood)
            angle_error = self._angle_diff(doa_angle, det.angle)
            angular_score = math.exp(-(angle_error / self.DOA_SIGMA) ** 2 / 2)

            # Depth confidence (closer = more reliable, falloff at extremes)
            depth_score = self._depth_confidence(det.z)

            # Combined score
            fusion_score = (
                self.ANGLE_WEIGHT * angular_score +
                self.DEPTH_WEIGHT * depth_score
            ) * speech_prob  # Modulate by speech probability

            if fusion_score > best_score:
                best_score = fusion_score
                best_match = SpeakerMatch(
                    detection=det,
                    doa_angle=doa_angle,
                    angle_error=abs(angle_error),
                    depth_confidence=depth_score,
                    fusion_score=fusion_score,
                    confidence=self._determine_confidence(angular_score, depth_score),
                    timestamp=time.time()
                )

        if best_match:
            return best_match
        else:
            # Detections exist but no people - DOA only
            return SpeakerMatch(
                detection=None,
                doa_angle=doa_angle,
                confidence=FusionConfidence.LOW,
                fusion_score=speech_prob * 0.5
            )

    def _angle_diff(self, a1: float, a2: float) -> float:
        """Compute signed angle difference, handling wraparound."""
        diff = a1 - a2
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def _depth_confidence(self, depth_mm: float) -> float:
        """Compute confidence based on depth reading.

        - Very close (<300mm): Low confidence (sensor limits)
        - Optimal (500-2000mm): High confidence
        - Far (>3000mm): Decreasing confidence
        - Invalid (0): No confidence
        """
        if depth_mm <= 0:
            return 0.0
        if depth_mm < self.MIN_DEPTH:
            return 0.3  # Too close
        if depth_mm < 500:
            return 0.7  # Getting close to optimal
        if depth_mm < 2000:
            return 1.0  # Optimal range
        if depth_mm < 3000:
            return 0.8  # Still good
        if depth_mm < self.MAX_DEPTH:
            return 0.5  # Far but usable
        return 0.2  # Beyond reliable range

    def _determine_confidence(self, angular_score: float, depth_score: float) -> FusionConfidence:
        """Determine confidence level from scores."""
        if angular_score > 0.8 and depth_score > 0.7:
            return FusionConfidence.HIGH
        elif angular_score > 0.5 and depth_score > 0.3:
            return FusionConfidence.MEDIUM
        elif angular_score > 0.3:
            return FusionConfidence.LOW
        return FusionConfidence.NONE

    def _is_match_changed(self, new_match: SpeakerMatch) -> bool:
        """Check if the speaker match has changed significantly."""
        if self._current_match is None:
            return new_match.is_valid

        old = self._current_match
        new = new_match

        # Different detection
        if old.detection is None and new.detection is not None:
            return True
        if old.detection is not None and new.detection is None:
            return True

        # Significant angle change
        if abs(old.doa_angle - new.doa_angle) > 20:
            return True

        # Confidence level change
        if old.confidence != new.confidence:
            return True

        return False

    def _notify_speaker_change(self, match: SpeakerMatch):
        """Notify callbacks of speaker change."""
        for cb in self._on_speaker_change:
            try:
                cb(match)
            except Exception as e:
                print(f"[DOASpatialFusion] Callback error: {e}")

    def get_active_speaker(self) -> Optional[SpeakerMatch]:
        """Get the current active speaker match.

        Returns:
            SpeakerMatch with detection and confidence, or None if no one speaking
        """
        with self._lock:
            return self._current_match

    def get_speaker_identity(self) -> Optional[Dict]:
        """Get identity of current speaker (if identity callback set).

        Returns:
            {'name': str, 'confidence': float, 'source': str} or None
        """
        if not self._identity_callback:
            return None

        with self._lock:
            if self._current_match is None or self._current_match.detection is None:
                return None
            return self._identity_callback(self._current_match.detection)

    def get_doa_angle(self) -> Tuple[float, bool]:
        """Get raw DOA angle (for gantry tracking).

        Returns:
            (angle_degrees, is_speaking)
        """
        if self._doa:
            angle, speaking, _ = self._doa.get_doa()
            return (angle, speaking)
        return (0.0, False)

    def get_speaker_position(self) -> Optional[Tuple[float, float, float]]:
        """Get 3D position of current speaker.

        Returns:
            (x, y, z) in millimeters, or None if no speaker detected
        """
        with self._lock:
            if self._current_match and self._current_match.detection:
                det = self._current_match.detection
                return (det.x, det.y, det.z)
        return None

    def get_speaker_distance(self) -> Optional[float]:
        """Get distance to current speaker in millimeters.

        Returns:
            Distance in mm, or None if no speaker detected
        """
        with self._lock:
            if self._current_match and self._current_match.detection:
                return self._current_match.detection.distance
        return None

    def enable_visual_safety(self, enabled: bool = True,
                              hazard_callback: Optional[Callable] = None) -> bool:
        """Enable visual safety detection (fire, smoke, etc).

        This feeds camera frames to the visual safety system for
        real-time hazard detection.

        Args:
            enabled: Whether to enable safety detection
            hazard_callback: Optional callback for hazard events
                            Receives HazardDetection with type, confidence, direction

        Returns:
            True if safety system was enabled successfully
        """
        if self._tracker:
            return self._tracker.enable_visual_safety(enabled, hazard_callback)
        else:
            print("[DOASpatialFusion] Tracker not started, cannot enable safety")
            return False


# =============================================================================
# Singleton
# =============================================================================
_fusion: Optional[DOASpatialFusion] = None
_fusion_lock = threading.Lock()


def get_fusion() -> DOASpatialFusion:
    """Get the singleton DOA-Spatial fusion instance."""
    global _fusion
    with _fusion_lock:
        if _fusion is None:
            _fusion = DOASpatialFusion()
        return _fusion


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("DOA + Spatial Fusion Test")
    print("=" * 50)
    print("Speak to see fusion in action...")
    print("Press Ctrl+C to exit\n")

    fusion = get_fusion()

    def on_speaker_change(match: SpeakerMatch):
        if match.detection:
            print(f"\n[SPEAKER CHANGED] {match.confidence.value.upper()} confidence")
            print(f"  DOA: {match.doa_angle:.1f}° | Detection: {match.detection.angle:.1f}°")
            print(f"  Distance: {match.detection.distance:.0f}mm")
            print(f"  Score: {match.fusion_score:.2f}")
        else:
            print(f"\n[NO VISUAL MATCH] DOA-only at {match.doa_angle:.1f}°")

    fusion.on_speaker_change(on_speaker_change)
    fusion.start()

    try:
        while True:
            match = fusion.get_active_speaker()
            if match and match.is_valid:
                det = match.detection
                print(f"Speaker: {det.distance/1000:.1f}m @ {det.angle:.0f}° "
                      f"[{match.confidence.value}] score={match.fusion_score:.2f}",
                      end='\r')
            elif match and match.doa_angle:
                print(f"Voice at {match.doa_angle:.0f}° (no visual match)          ",
                      end='\r')
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        fusion.stop()
        print("Test complete!")
