"""Base Microphone Array Interface

Abstract interface for microphone arrays with DOA.
Implement this for each microphone platform.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable, Tuple
from enum import Enum
import threading


class MicrophoneType(Enum):
    """Supported microphone array types."""
    RESPEAKER_4MIC = "respeaker_4mic"
    RESPEAKER_6MIC = "respeaker_6mic"
    CIRCULAR_6MIC = "circular_6mic"
    UNITREE_4MIC = "unitree_4mic"
    MATRIX_VOICE = "matrix_voice"
    ODAS = "odas"  # Generic ODAS-based array


@dataclass
class DOAReading:
    """A single DOA reading with metadata."""
    angle: float  # Degrees (0-360)
    confidence: float  # 0.0 - 1.0
    is_speaking: bool
    energy: float = 0.0  # Audio energy level
    timestamp: float = 0.0


class MicrophoneArray(ABC):
    """Abstract interface for microphone arrays.

    Implement this for each microphone platform to enable
    the same DOA tracking code to work across different hardware.

    Example:
        mic = Circular6Mic()
        mic.start()

        doa, speaking, conf = mic.get_doa()
        print(f"Sound from {doa}° (speaking: {speaking})")
    """

    def __init__(self):
        self._running = False
        self._callbacks: List[Callable[[float, bool], None]] = []
        self._smoothed_doa: float = 0.0
        self._is_speaking: bool = False
        self._confidence: float = 0.0
        self._lock = threading.Lock()

    @property
    @abstractmethod
    def mic_type(self) -> MicrophoneType:
        """Return the microphone type."""
        pass

    @property
    @abstractmethod
    def num_mics(self) -> int:
        """Return number of microphones in array."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return audio sample rate in Hz."""
        pass

    @property
    def doa_resolution(self) -> float:
        """Return DOA resolution in degrees.

        More mics = better resolution.
        """
        # Rule of thumb: 360 / (num_mics * 3)
        return 360.0 / (self.num_mics * 3)

    @abstractmethod
    def start(self, poll_rate_hz: int = 30) -> bool:
        """Start DOA tracking.

        Args:
            poll_rate_hz: How often to update DOA

        Returns:
            True if started successfully
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop DOA tracking."""
        pass

    @abstractmethod
    def read_doa_raw(self) -> Optional[int]:
        """Read raw DOA from hardware.

        Returns:
            DOA angle in degrees (0-360), or None
        """
        pass

    @abstractmethod
    def read_vad(self) -> bool:
        """Read Voice Activity Detection status.

        Returns:
            True if voice is detected
        """
        pass

    def get_doa(self) -> Tuple[float, bool, float]:
        """Get current smoothed DOA and speaking status.

        This is the recommended method - uses smoothed values.

        Returns:
            (doa_degrees, is_speaking, confidence)
        """
        with self._lock:
            return (self._smoothed_doa, self._is_speaking, self._confidence)

    def on_direction_change(self, callback: Callable[[float, bool], None]):
        """Register callback for direction changes.

        Callback receives (doa_degrees, is_speaking).
        """
        self._callbacks.append(callback)

    def doa_to_pan(self, doa: float, offset: float = 0.0) -> float:
        """Convert DOA to robot pan angle.

        Args:
            doa: DOA from array (0-360 degrees)
            offset: Mounting offset if not facing forward

        Returns:
            Pan angle (-90 to +90, clamped)
        """
        adjusted = (doa - offset) % 360

        if adjusted > 180:
            adjusted -= 360

        return max(-90, min(90, adjusted))

    @property
    def is_running(self) -> bool:
        """Check if tracking is running."""
        return self._running

    def _update_doa(self, raw_doa: float, vad: bool, confidence: float = 1.0):
        """Update internal state with new reading.

        Called by implementations to update smoothed DOA.
        """
        with self._lock:
            self._is_speaking = vad
            self._confidence = confidence

            if vad and raw_doa is not None:
                # Exponential smoothing with wraparound handling
                diff = raw_doa - self._smoothed_doa
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360

                smoothing = 0.3
                self._smoothed_doa += smoothing * diff
                self._smoothed_doa %= 360

    def _fire_callbacks(self, doa: float, speaking: bool):
        """Fire registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(doa, speaking)
            except Exception as e:
                print(f"[MicrophoneArray] Callback error: {e}")
