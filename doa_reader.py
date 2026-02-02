#!/usr/bin/env python3
"""ReSpeaker DOA (Direction of Arrival) Reader

Reads the direction of arrival from the ReSpeaker USB Mic Array.
The XMOS XVF-3000 DSP computes DOA from the 4-mic array using
onboard beamforming algorithms.

The DOA is returned as degrees (0-360), where:
- 0° = Front of device
- 90° = Right
- 180° = Back
- 270° = Left

This value is updated in real-time by the firmware whenever
voice activity is detected.
"""

import time
import threading
from typing import Optional, Callable, Tuple

try:
    import usb.core
    import usb.util
    USB_AVAILABLE = True
except ImportError:
    USB_AVAILABLE = False
    print("WARNING: pyusb not installed. Run: pip install pyusb --break-system-packages")


class ReSpeakerDOA:
    """Read Direction of Arrival from ReSpeaker 4-Mic Array.

    The ReSpeaker provides:
    - DOA angle (0-360°)
    - Voice Activity Detection (VAD) status
    - Audio levels per microphone
    """

    VENDOR_ID = 0x2886   # Seeed Studio
    PRODUCT_ID = 0x0018  # ReSpeaker 4-Mic Array

    # XMOS XVF-3000 Tuning registers for DOA
    # These are read via USB control transfers
    REGISTER_DOA = 0x21          # Direction of arrival (0-360)
    REGISTER_VAD = 0x20          # Voice activity detection
    REGISTER_MIC_LEVELS = 0x22   # Individual mic levels
    REGISTER_SPEECH_PROB = 0x23  # Speech probability (0-255)

    def __init__(self, smoothing: float = 0.3):
        """Initialize DOA reader.

        Args:
            smoothing: Exponential smoothing factor (0-1).
                       Higher = more responsive, noisier
                       Lower = smoother, slower
        """
        self.dev = None
        self.smoothing = smoothing
        self._current_doa: float = 0.0
        self._smoothed_doa: float = 0.0
        self._is_speaking: bool = False
        self._speech_prob: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list = []

        if not USB_AVAILABLE:
            print("DOA Reader: USB not available, running in dummy mode")
            return

        try:
            self.dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
            if self.dev is None:
                print("DOA Reader: ReSpeaker not found on USB")
            else:
                # Detach kernel driver if attached
                if self.dev.is_kernel_driver_active(0):
                    try:
                        self.dev.detach_kernel_driver(0)
                    except usb.core.USBError:
                        pass
                print("DOA Reader: ReSpeaker found and ready")
        except Exception as e:
            print(f"DOA Reader: USB init error: {e}")
            self.dev = None

    def _read_register(self, register: int, length: int = 2) -> Optional[bytes]:
        """Read a value from an XMOS register."""
        if not self.dev:
            return None

        try:
            data = self.dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,  # bRequest
                0,  # wValue
                register,  # wIndex
                length,  # data length
                timeout=100
            )
            return bytes(data)
        except Exception as e:
            # Timeout or error - common during silence
            return None

    def read_doa(self) -> Optional[int]:
        """Read current DOA angle from firmware.

        Returns:
            DOA angle in degrees (0-360), or None if unavailable
        """
        data = self._read_register(self.REGISTER_DOA, 2)
        if data and len(data) >= 2:
            # DOA is typically a 16-bit value
            doa = int.from_bytes(data[:2], 'little')
            # Some firmware returns values 0-3600 (tenths of degrees)
            if doa > 360:
                doa = doa // 10
            return doa % 360
        return None

    def read_vad(self) -> bool:
        """Read Voice Activity Detection status.

        Returns:
            True if voice is detected, False otherwise
        """
        data = self._read_register(self.REGISTER_VAD, 1)
        if data:
            return data[0] > 0
        return False

    def read_speech_probability(self) -> float:
        """Read speech probability (0.0 - 1.0).

        Returns:
            Probability that current audio contains speech
        """
        data = self._read_register(self.REGISTER_SPEECH_PROB, 1)
        if data:
            return data[0] / 255.0
        return 0.0

    def read_mic_levels(self) -> Optional[Tuple[int, int, int, int]]:
        """Read individual microphone levels.

        Returns:
            Tuple of 4 mic levels (0-255), or None if unavailable
        """
        data = self._read_register(self.REGISTER_MIC_LEVELS, 4)
        if data and len(data) >= 4:
            return (data[0], data[1], data[2], data[3])
        return None

    def get_doa(self) -> Tuple[float, bool, float]:
        """Get the current smoothed DOA and speaking status.

        This is the recommended method for tracking - it uses
        smoothed values from the background thread.

        Returns:
            (doa_degrees, is_speaking, speech_probability)
        """
        with self._lock:
            return (self._smoothed_doa, self._is_speaking, self._speech_prob)

    def on_direction_change(self, callback: Callable[[float, bool], None]):
        """Register callback for direction changes.

        Callback receives (doa_degrees, is_speaking) when someone
        starts speaking or direction changes significantly.
        """
        self._callbacks.append(callback)

    def start(self, poll_rate_hz: int = 30):
        """Start background DOA tracking.

        Args:
            poll_rate_hz: How often to read DOA (default 30Hz)
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, args=(poll_rate_hz,), daemon=True)
        self._thread.start()
        print(f"DOA Reader: Started at {poll_rate_hz}Hz")

    def stop(self):
        """Stop background DOA tracking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        print("DOA Reader: Stopped")

    def _poll_loop(self, poll_rate_hz: int):
        """Background thread that continuously reads DOA."""
        period = 1.0 / poll_rate_hz
        last_speaking = False
        last_doa = 0.0

        while self._running:
            loop_start = time.perf_counter()

            # Read all values
            raw_doa = self.read_doa()
            vad = self.read_vad()
            speech_prob = self.read_speech_probability()

            with self._lock:
                self._is_speaking = vad
                self._speech_prob = speech_prob

                if raw_doa is not None and vad:
                    # Only update DOA when voice is detected
                    self._current_doa = raw_doa

                    # Apply exponential smoothing
                    # Handle wraparound (e.g., 350° -> 10°)
                    diff = raw_doa - self._smoothed_doa
                    if diff > 180:
                        diff -= 360
                    elif diff < -180:
                        diff += 360

                    self._smoothed_doa += self.smoothing * diff
                    self._smoothed_doa %= 360

            # Fire callbacks if state changed significantly
            doa_changed = abs(self._smoothed_doa - last_doa) > 15  # 15° threshold
            speaking_changed = vad != last_speaking

            if (speaking_changed or (vad and doa_changed)) and self._callbacks:
                for cb in self._callbacks:
                    try:
                        cb(self._smoothed_doa, vad)
                    except Exception as e:
                        print(f"DOA callback error: {e}")

                last_doa = self._smoothed_doa
                last_speaking = vad

            # Maintain poll rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def doa_to_robot_pan(self, doa: float, robot_forward: float = 0.0) -> float:
        """Convert DOA to robot gantry pan angle.

        The ReSpeaker reports DOA relative to its front (0°).
        This converts to the robot's pan servo coordinate.

        Args:
            doa: DOA from ReSpeaker (0-360°)
            robot_forward: Angle offset if ReSpeaker isn't mounted
                          facing robot forward (default 0)

        Returns:
            Pan angle for gantry (-90 to +90, clamped)
        """
        # Adjust for mounting orientation
        adjusted = (doa - robot_forward) % 360

        # Convert to signed angle (-180 to +180)
        if adjusted > 180:
            adjusted -= 360

        # ReSpeaker uses clockwise positive (looking down)
        # Robot pan is typically: negative = left, positive = right
        # Invert if needed based on your mount
        pan = adjusted

        # Clamp to gantry limits
        pan = max(-90, min(90, pan))

        return pan


# Singleton instance
_doa_instance: Optional[ReSpeakerDOA] = None
_doa_lock = threading.Lock()


def get_doa() -> ReSpeakerDOA:
    """Get the singleton DOA reader instance."""
    global _doa_instance
    with _doa_lock:
        if _doa_instance is None:
            _doa_instance = ReSpeakerDOA()
        return _doa_instance


# Quick test
if __name__ == "__main__":
    print("Testing ReSpeaker DOA Reader")
    print("=" * 50)
    print("Speak to see direction detection...")
    print("Press Ctrl+C to exit\n")

    doa = get_doa()

    def on_direction(angle: float, speaking: bool):
        status = "SPEAKING" if speaking else "silent"
        print(f"  → Direction: {angle:5.1f}° [{status}]")

    doa.on_direction_change(on_direction)
    doa.start(poll_rate_hz=30)

    try:
        while True:
            angle, speaking, prob = doa.get_doa()
            if speaking:
                pan = doa.doa_to_robot_pan(angle)
                print(f"DOA: {angle:5.1f}° | Pan: {pan:+5.1f}° | Prob: {prob:.0%}", end='\r')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        doa.stop()
        print("Test complete!")
