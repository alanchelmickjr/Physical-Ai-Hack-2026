"""ReSpeaker Microphone Array Implementation

Supports ReSpeaker 4-Mic and 6-Mic arrays.
Uses USB control transfers to read DOA from XMOS firmware.
"""

import time
import threading
from typing import Optional, Tuple

from .base import MicrophoneArray, MicrophoneType

try:
    import usb.core
    import usb.util
    USB_AVAILABLE = True
except ImportError:
    USB_AVAILABLE = False
    print("WARNING: pyusb not installed. Run: pip install pyusb --break-system-packages")


class ReSpeakerMic(MicrophoneArray):
    """ReSpeaker 4-Mic or 6-Mic Array.

    Uses the XMOS XVF-3000 DSP for onboard DOA computation.
    DOA is read via USB control transfers.
    """

    # USB identifiers
    VENDOR_ID = 0x2886  # Seeed Studio

    # Product IDs
    PRODUCT_IDS = {
        MicrophoneType.RESPEAKER_4MIC: 0x0018,
        MicrophoneType.RESPEAKER_6MIC: 0x0018,  # Same for 6-mic USB version
    }

    # XMOS registers
    REGISTER_DOA = 0x21
    REGISTER_VAD = 0x20
    REGISTER_MIC_LEVELS = 0x22
    REGISTER_SPEECH_PROB = 0x23

    def __init__(self, mic_type: MicrophoneType = MicrophoneType.RESPEAKER_4MIC):
        """Initialize ReSpeaker array.

        Args:
            mic_type: RESPEAKER_4MIC or RESPEAKER_6MIC
        """
        super().__init__()
        self._type = mic_type
        self._num_mics = 4 if mic_type == MicrophoneType.RESPEAKER_4MIC else 6
        self._dev = None
        self._thread: Optional[threading.Thread] = None

        if USB_AVAILABLE:
            self._init_usb()

    def _init_usb(self):
        """Initialize USB connection."""
        try:
            product_id = self.PRODUCT_IDS.get(self._type, 0x0018)
            self._dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=product_id)

            if self._dev is None:
                print(f"[ReSpeaker] Device not found on USB")
            else:
                if self._dev.is_kernel_driver_active(0):
                    try:
                        self._dev.detach_kernel_driver(0)
                    except usb.core.USBError:
                        pass
                print(f"[ReSpeaker] Found {self._type.value} on USB")

        except Exception as e:
            print(f"[ReSpeaker] USB init error: {e}")

    @property
    def mic_type(self) -> MicrophoneType:
        return self._type

    @property
    def num_mics(self) -> int:
        return self._num_mics

    @property
    def sample_rate(self) -> int:
        return 16000

    def _read_register(self, register: int, length: int = 2) -> Optional[bytes]:
        """Read from XMOS register."""
        if not self._dev:
            return None

        try:
            data = self._dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,
                0,
                register,
                length,
                timeout=100
            )
            return bytes(data)
        except Exception:
            return None

    def read_doa_raw(self) -> Optional[int]:
        """Read raw DOA from firmware."""
        data = self._read_register(self.REGISTER_DOA, 2)
        if data and len(data) >= 2:
            doa = int.from_bytes(data[:2], 'little')
            if doa > 360:
                doa = doa // 10
            return doa % 360
        return None

    def read_vad(self) -> bool:
        """Read VAD status."""
        data = self._read_register(self.REGISTER_VAD, 1)
        if data:
            return data[0] > 0
        return False

    def read_speech_probability(self) -> float:
        """Read speech probability (0.0 - 1.0)."""
        data = self._read_register(self.REGISTER_SPEECH_PROB, 1)
        if data:
            return data[0] / 255.0
        return 0.0

    def read_mic_levels(self) -> Optional[Tuple]:
        """Read individual mic levels."""
        data = self._read_register(self.REGISTER_MIC_LEVELS, self._num_mics)
        if data and len(data) >= self._num_mics:
            return tuple(data[:self._num_mics])
        return None

    def start(self, poll_rate_hz: int = 30) -> bool:
        """Start DOA tracking."""
        if not USB_AVAILABLE or not self._dev:
            print("[ReSpeaker] Cannot start - device not available")
            return False

        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(poll_rate_hz,),
            daemon=True
        )
        self._thread.start()
        print(f"[ReSpeaker] Started at {poll_rate_hz}Hz")
        return True

    def stop(self):
        """Stop DOA tracking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        print("[ReSpeaker] Stopped")

    def _poll_loop(self, poll_rate_hz: int):
        """Background polling loop."""
        period = 1.0 / poll_rate_hz
        last_speaking = False
        last_doa = 0.0

        while self._running:
            loop_start = time.perf_counter()

            raw_doa = self.read_doa_raw()
            vad = self.read_vad()
            prob = self.read_speech_probability()

            self._update_doa(raw_doa or 0, vad, prob)

            # Fire callbacks on significant change
            doa_changed = abs(self._smoothed_doa - last_doa) > 15
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
