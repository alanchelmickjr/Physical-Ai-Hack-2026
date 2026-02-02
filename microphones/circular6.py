"""Circular 6-Microphone Array Implementation

Generic implementation for 6-mic circular arrays.
Uses pyroomacoustics for DOA estimation when hardware doesn't provide it.

Used by: Booster K1
"""

import time
import threading
from typing import Optional, List, Tuple
import numpy as np

from .base import MicrophoneArray, MicrophoneType

try:
    import pyroomacoustics as pra
    PRA_AVAILABLE = True
except ImportError:
    PRA_AVAILABLE = False
    print("WARNING: pyroomacoustics not installed. Run: pip install pyroomacoustics")

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    print("WARNING: sounddevice not installed. Run: pip install sounddevice")

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False


class Circular6Mic(MicrophoneArray):
    """Circular 6-Microphone Array with SRP-PHAT DOA.

    Computes DOA using SRP-PHAT algorithm from pyroomacoustics.
    Works with any 6-mic circular array (Booster K1, custom arrays).

    Features:
    - SRP-PHAT DOA estimation
    - Silero VAD for voice activity detection
    - Configurable array geometry
    """

    def __init__(
        self,
        mic_type: MicrophoneType = MicrophoneType.CIRCULAR_6MIC,
        radius_mm: float = 45.0,
        device_index: Optional[int] = None,
    ):
        """Initialize circular 6-mic array.

        Args:
            mic_type: Microphone type enum
            radius_mm: Array radius in millimeters
            device_index: Audio device index (None = default)
        """
        super().__init__()
        self._type = mic_type
        self._radius = radius_mm / 1000.0  # Convert to meters
        self._device_index = device_index
        self._sample_rate = 16000

        # Mic positions (circular, 60 degree spacing)
        self._mic_positions = self._compute_mic_positions()

        # Audio buffer
        self._buffer_size = 1024
        self._audio_buffer: Optional[np.ndarray] = None

        # DOA estimator
        self._doa_estimator = None

        # VAD
        self._vad_model = None
        if VAD_AVAILABLE:
            try:
                self._vad_model = load_silero_vad(onnx=True)
            except Exception as e:
                print(f"[Circular6Mic] VAD load failed: {e}")

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stream = None

    def _compute_mic_positions(self) -> np.ndarray:
        """Compute microphone positions in 3D.

        Returns:
            Array of shape (3, 6) with x, y, z coordinates
        """
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 mics, 60 degrees apart
        x = self._radius * np.cos(angles)
        y = self._radius * np.sin(angles)
        z = np.zeros(6)
        return np.array([x, y, z])

    @property
    def mic_type(self) -> MicrophoneType:
        return self._type

    @property
    def num_mics(self) -> int:
        return 6

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def read_doa_raw(self) -> Optional[int]:
        """Read DOA from audio buffer using SRP-PHAT.

        Returns:
            DOA in degrees (0-360), or None
        """
        if self._audio_buffer is None or not PRA_AVAILABLE:
            return None

        try:
            # Create DOA estimator if needed
            if self._doa_estimator is None:
                self._doa_estimator = pra.doa.SRP(
                    self._mic_positions,
                    self._sample_rate,
                    self._buffer_size,
                    c=343.0,  # Speed of sound
                    num_src=1,
                    mode='far',
                )

            # Ensure buffer is correct shape (channels, samples)
            if self._audio_buffer.ndim == 1:
                return None
            if self._audio_buffer.shape[0] != 6:
                self._audio_buffer = self._audio_buffer.T

            # Estimate DOA
            self._doa_estimator.locate_sources(self._audio_buffer)

            if len(self._doa_estimator.azimuth_recon) > 0:
                # Convert from radians to degrees
                azimuth_rad = self._doa_estimator.azimuth_recon[0]
                azimuth_deg = np.degrees(azimuth_rad)

                # Convert to 0-360 range
                if azimuth_deg < 0:
                    azimuth_deg += 360

                return int(azimuth_deg)

        except Exception as e:
            print(f"[Circular6Mic] DOA error: {e}")

        return None

    def read_vad(self) -> bool:
        """Detect voice activity using Silero VAD."""
        if self._audio_buffer is None:
            return False

        # Use energy-based VAD as fallback
        if self._vad_model is None:
            energy = np.mean(np.abs(self._audio_buffer))
            return energy > 0.01  # Threshold

        try:
            # Silero VAD expects mono audio
            if self._audio_buffer.ndim > 1:
                mono = np.mean(self._audio_buffer, axis=0)
            else:
                mono = self._audio_buffer

            # Get speech probability
            import torch
            audio_tensor = torch.from_numpy(mono).float()
            speech_prob = self._vad_model(audio_tensor, self._sample_rate).item()

            return speech_prob > 0.5

        except Exception:
            # Fallback to energy
            energy = np.mean(np.abs(self._audio_buffer))
            return energy > 0.01

    def start(self, poll_rate_hz: int = 30) -> bool:
        """Start DOA tracking."""
        if not SD_AVAILABLE:
            print("[Circular6Mic] sounddevice not available")
            return False

        if not PRA_AVAILABLE:
            print("[Circular6Mic] pyroomacoustics not available")
            return False

        if self._running:
            return True

        try:
            # Find 6-channel input device
            if self._device_index is None:
                self._device_index = self._find_device()

            if self._device_index is None:
                print("[Circular6Mic] No 6-channel device found")
                return False

            self._running = True

            # Start audio stream
            self._stream = sd.InputStream(
                device=self._device_index,
                channels=6,
                samplerate=self._sample_rate,
                blocksize=self._buffer_size,
                callback=self._audio_callback,
            )
            self._stream.start()

            # Start processing thread
            self._thread = threading.Thread(
                target=self._process_loop,
                args=(poll_rate_hz,),
                daemon=True
            )
            self._thread.start()

            print(f"[Circular6Mic] Started at {poll_rate_hz}Hz")
            return True

        except Exception as e:
            print(f"[Circular6Mic] Start failed: {e}")
            self._running = False
            return False

    def stop(self):
        """Stop DOA tracking."""
        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        print("[Circular6Mic] Stopped")

    def _find_device(self) -> Optional[int]:
        """Find a 6-channel input device."""
        devices = sd.query_devices()

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] >= 6:
                print(f"[Circular6Mic] Using device {i}: {dev['name']}")
                return i

        return None

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        if status:
            print(f"[Circular6Mic] Audio status: {status}")

        # Store buffer (shape: samples x channels)
        self._audio_buffer = indata.T.copy()  # Convert to channels x samples

    def _process_loop(self, poll_rate_hz: int):
        """Background processing loop."""
        period = 1.0 / poll_rate_hz
        last_speaking = False
        last_doa = 0.0

        while self._running:
            loop_start = time.perf_counter()

            if self._audio_buffer is not None:
                raw_doa = self.read_doa_raw()
                vad = self.read_vad()

                self._update_doa(raw_doa or 0, vad, 0.8 if vad else 0.2)

                # Fire callbacks on significant change
                doa_changed = abs(self._smoothed_doa - last_doa) > 12  # 12 deg threshold
                speaking_changed = vad != last_speaking

                if (speaking_changed or (vad and doa_changed)):
                    self._fire_callbacks(self._smoothed_doa, vad)
                    last_doa = self._smoothed_doa
                    last_speaking = vad

            # Maintain poll rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_audio_levels(self) -> Optional[Tuple[float, ...]]:
        """Get audio levels per microphone.

        Returns:
            Tuple of 6 RMS levels, or None
        """
        if self._audio_buffer is None:
            return None

        try:
            # RMS per channel
            rms = np.sqrt(np.mean(self._audio_buffer ** 2, axis=1))
            return tuple(rms)
        except Exception:
            return None
