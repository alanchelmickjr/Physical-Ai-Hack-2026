#!/usr/bin/env python3
"""Head Tracker - Makes Chloe look at who's talking

This module tracks audio direction and controls the gantry
to keep the camera pointed at the speaker.

Pipeline:
1. DOA from ReSpeaker → sound direction
2. Convert to gantry pan angle
3. Move head smoothly toward speaker
4. If speaker is outside gantry range, signal base rotation

Features:
- Smooth pursuit tracking (not jerky)
- Dead zone to avoid micro-movements
- Field-of-view awareness (camera FOV vs gantry range)
- Out-of-range detection for base rotation
"""

import asyncio
import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from enum import Enum

from doa_reader import ReSpeakerDOA, get_doa


class SpeakerLocation(Enum):
    """Where the speaker is relative to Chloe's view."""
    IN_VIEW = "in_view"           # Within camera FOV
    GANTRY_REACHABLE = "gantry"   # Outside FOV but gantry can reach
    NEED_BASE_TURN = "base"       # Need to rotate the base
    BEHIND = "behind"             # Behind the robot


@dataclass
class TrackingConfig:
    """Configuration for head tracking behavior."""
    # Gantry mechanical limits
    pan_min: float = -90.0   # degrees
    pan_max: float = 90.0

    tilt_min: float = -45.0
    tilt_max: float = 45.0

    # Camera field of view
    camera_fov_h: float = 72.0   # OAK-D horizontal FOV
    camera_fov_v: float = 50.0

    # Tracking behavior
    dead_zone: float = 5.0       # Ignore movements smaller than this
    smooth_factor: float = 0.2   # Tracking smoothness (0-1)
    track_speed: float = 0.5     # Motor speed (0-1)

    # Timing
    settle_time: float = 0.5     # Pause after reaching target
    min_track_interval: float = 0.1  # Minimum time between moves

    # ReSpeaker mounting offset (degrees)
    # Set this if ReSpeaker isn't facing robot forward
    respeaker_offset: float = 0.0


@dataclass
class SpeakerState:
    """Current state of the detected speaker."""
    doa: float = 0.0              # Raw DOA from ReSpeaker
    pan_target: float = 0.0       # Target pan angle
    tilt_target: float = 0.0      # Target tilt (default center)
    is_speaking: bool = False
    speech_prob: float = 0.0
    location: SpeakerLocation = SpeakerLocation.IN_VIEW
    last_seen: float = 0.0        # Time of last voice detection


class HeadTracker:
    """Controls Chloe's head to track speakers.

    Uses DOA from ReSpeaker to control the 2-DOF gantry,
    keeping the camera pointed at whoever is speaking.
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        doa_reader: Optional[ReSpeakerDOA] = None
    ):
        self.config = config or TrackingConfig()
        self.doa = doa_reader or get_doa()

        self._state = SpeakerState()
        self._current_pan: float = 0.0
        self._current_tilt: float = 0.0
        self._target_pan: float = 0.0
        self._target_tilt: float = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Callbacks
        self._on_speaker_detected: Optional[Callable] = None
        self._on_out_of_range: Optional[Callable] = None
        self._on_move_gantry: Optional[Callable] = None

        # Motor control function - set via set_gantry_controller
        self._move_gantry: Optional[Callable[[float, float, float], None]] = None

    def set_gantry_controller(self, move_fn: Callable[[float, float, float], None]):
        """Set the function to move the gantry.

        Args:
            move_fn: Function(pan, tilt, speed) that moves gantry
        """
        self._move_gantry = move_fn

    def on_speaker_detected(self, callback: Callable[[SpeakerState], None]):
        """Register callback when speaker is detected."""
        self._on_speaker_detected = callback

    def on_out_of_range(self, callback: Callable[[float, SpeakerLocation], None]):
        """Register callback when speaker is outside gantry range.

        This signals that base rotation is needed.
        callback(doa_angle, location_type)
        """
        self._on_out_of_range = callback

    def _doa_to_location(self, doa: float) -> Tuple[float, SpeakerLocation]:
        """Convert DOA to pan target and determine location.

        Returns:
            (pan_angle, speaker_location)
        """
        # Adjust for ReSpeaker mounting
        adjusted = (doa - self.config.respeaker_offset) % 360

        # Convert to signed angle (-180 to +180)
        if adjusted > 180:
            adjusted -= 360

        # Determine where the speaker is
        pan = adjusted

        # Within camera FOV (no head movement needed)
        half_fov = self.config.camera_fov_h / 2
        if abs(pan - self._current_pan) <= half_fov:
            return pan, SpeakerLocation.IN_VIEW

        # Within gantry range
        if self.config.pan_min <= pan <= self.config.pan_max:
            return pan, SpeakerLocation.GANTRY_REACHABLE

        # Behind the robot (need 180° turn)
        if abs(pan) > 135:
            return pan, SpeakerLocation.BEHIND

        # Need base rotation
        return pan, SpeakerLocation.NEED_BASE_TURN

    def _should_move(self, target_pan: float) -> bool:
        """Check if we should move to a new target."""
        diff = abs(target_pan - self._current_pan)
        return diff > self.config.dead_zone

    def start(self, poll_rate_hz: int = 30):
        """Start head tracking.

        Args:
            poll_rate_hz: Tracking update rate
        """
        if self._running:
            return

        # Start DOA reader if not running
        self.doa.start(poll_rate_hz=poll_rate_hz)

        # Register DOA callback
        self.doa.on_direction_change(self._on_doa_change)

        self._running = True
        self._thread = threading.Thread(target=self._track_loop, daemon=True)
        self._thread.start()

        print(f"Head Tracker: Started at {poll_rate_hz}Hz")

    def stop(self):
        """Stop head tracking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.doa.stop()
        print("Head Tracker: Stopped")

    def _on_doa_change(self, doa: float, speaking: bool):
        """Handle DOA changes from ReSpeaker."""
        if not speaking:
            return

        with self._lock:
            pan_target, location = self._doa_to_location(doa)

            self._state.doa = doa
            self._state.pan_target = pan_target
            self._state.is_speaking = speaking
            self._state.location = location
            self._state.last_seen = time.time()

            # Notify callbacks
            if self._on_speaker_detected:
                self._on_speaker_detected(self._state)

            # If speaker is out of gantry range, signal for base rotation
            if location in (SpeakerLocation.NEED_BASE_TURN, SpeakerLocation.BEHIND):
                if self._on_out_of_range:
                    self._on_out_of_range(doa, location)

    def _track_loop(self):
        """Main tracking loop."""
        last_move_time = 0

        while self._running:
            with self._lock:
                state = SpeakerState(
                    doa=self._state.doa,
                    pan_target=self._state.pan_target,
                    tilt_target=self._state.tilt_target,
                    is_speaking=self._state.is_speaking,
                    location=self._state.location,
                    last_seen=self._state.last_seen
                )

            now = time.time()

            # Check if we should track
            if (state.is_speaking and
                state.location == SpeakerLocation.GANTRY_REACHABLE and
                self._should_move(state.pan_target) and
                now - last_move_time > self.config.min_track_interval):

                # Calculate smooth target
                diff = state.pan_target - self._current_pan
                smooth_target = self._current_pan + diff * self.config.smooth_factor

                # Clamp to limits
                smooth_target = max(self.config.pan_min,
                                   min(self.config.pan_max, smooth_target))

                # Move gantry if controller is set
                if self._move_gantry:
                    self._move_gantry(
                        smooth_target,
                        self._current_tilt,
                        self.config.track_speed
                    )
                    self._current_pan = smooth_target

                last_move_time = now

            time.sleep(0.03)  # ~30Hz internal loop

    def get_state(self) -> SpeakerState:
        """Get current speaker tracking state."""
        with self._lock:
            return SpeakerState(
                doa=self._state.doa,
                pan_target=self._state.pan_target,
                tilt_target=self._state.tilt_target,
                is_speaking=self._state.is_speaking,
                location=self._state.location,
                last_seen=self._state.last_seen
            )

    def look_at_angle(self, pan: float, tilt: float = 0.0, speed: float = 0.5):
        """Manually point the head at an angle.

        Args:
            pan: Horizontal angle (-90 to +90)
            tilt: Vertical angle (-45 to +45)
            speed: Movement speed (0-1)
        """
        # Clamp to limits
        pan = max(self.config.pan_min, min(self.config.pan_max, pan))
        tilt = max(self.config.tilt_min, min(self.config.tilt_max, tilt))

        if self._move_gantry:
            self._move_gantry(pan, tilt, speed)
            self._current_pan = pan
            self._current_tilt = tilt

    def center(self, speed: float = 0.5):
        """Center the head (look forward)."""
        self.look_at_angle(0.0, 0.0, speed)


# Singleton instance
_tracker_instance: Optional[HeadTracker] = None
_tracker_lock = threading.Lock()


def get_head_tracker() -> HeadTracker:
    """Get the singleton head tracker instance."""
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = HeadTracker()
        return _tracker_instance


# Quick test
if __name__ == "__main__":
    import subprocess
    import asyncio

    print("Testing Head Tracker")
    print("=" * 50)
    print("Speak from different directions...")
    print("Press Ctrl+C to exit\n")

    tracker = get_head_tracker()

    # Mock gantry controller (prints instead of moving)
    def mock_move_gantry(pan: float, tilt: float, speed: float):
        print(f"  GANTRY → Pan: {pan:+6.1f}° | Tilt: {tilt:+6.1f}° | Speed: {speed:.1f}")

    # Real gantry controller using Solo-CLI
    def solo_move_gantry(pan: float, tilt: float, speed: float):
        # Gantry is motors 7 (pan), 8 (tilt) on /dev/ttyACM1
        cmd = [
            "solo", "robo",
            "--port", "/dev/ttyACM1",
            "--ids", "7,8",
            "--positions", f"{pan},{tilt}",
            "--speed", str(speed)
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=1.0)
            print(f"  GANTRY → Pan: {pan:+6.1f}° | Tilt: {tilt:+6.1f}°")
        except Exception as e:
            print(f"  GANTRY ERROR: {e}")

    # Use mock for testing, swap to solo_move_gantry for real hardware
    tracker.set_gantry_controller(mock_move_gantry)

    def on_speaker(state: SpeakerState):
        loc = state.location.value
        print(f"  Speaker at {state.doa:5.1f}° → Target pan: {state.pan_target:+6.1f}° [{loc}]")

    def on_out_of_range(doa: float, location: SpeakerLocation):
        if location == SpeakerLocation.BEHIND:
            print(f"  !!! SPEAKER BEHIND ({doa:.0f}°) - Need 180° turn !!!")
        else:
            print(f"  !! Speaker at {doa:.0f}° - Need base rotation !!")

    tracker.on_speaker_detected(on_speaker)
    tracker.on_out_of_range(on_out_of_range)

    tracker.start(poll_rate_hz=30)

    try:
        while True:
            state = tracker.get_state()
            status = "SPEAKING" if state.is_speaking else "silent"
            print(f"DOA: {state.doa:5.1f}° | Pan: {state.pan_target:+6.1f}° | {status}   ", end='\r')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        tracker.stop()
        print("Test complete!")
