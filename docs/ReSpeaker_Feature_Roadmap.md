# ReSpeaker Feature Roadmap for Johnny Five

## Current State Summary

**What's Working:**
- Basic audio capture via `sounddevice` with auto-detection of ReSpeaker
- PulseAudio WebRTC echo cancellation (noise suppression + digital gain control)
- Mic muting during playback to prevent feedback loops
- Hume EVI voice conversation pipeline
- Face recognition triggers voice greetings via file IPC

**What's Untapped:**
| Feature | Hardware Capability | Current Usage |
|---------|---------------------|---------------|
| 12 RGB LEDs | Programmable status indicators | None |
| DOA (Direction of Arrival) | Real-time angle from 4-mic array | None |
| Raw 6-Channel Audio | 4 mics + 2 processed channels | Only 1-ch processed |
| Firmware VAD | On-chip voice activity detection | Using Hume's VAD |
| Speaker Identification | WeSpeaker embeddings from voice | Planned, not implemented |
| Beamforming Control | Steerable audio focus | Hardware default only |

---

## Priority 1: LED Ring Feedback (Quick Win)

**Why:** Immediate visual feedback makes the robot feel alive. Users know when Johnny is listening, thinking, or speaking.

**Library:** `pixel_ring` (ReSpeaker-specific) or direct USB HID control

**Installation:**
```bash
pip install pixel_ring --break-system-packages
```

**Implementation:**
```python
from pixel_ring import pixel_ring

# LED States for Johnny Five
def led_listening():
    """Blue spinning animation - Johnny is listening"""
    pixel_ring.spin()

def led_thinking():
    """Purple pulse - Processing/thinking"""
    pixel_ring.think()

def led_speaking():
    """Green solid - Johnny is talking"""
    pixel_ring.speak()

def led_off():
    """Turn off LEDs"""
    pixel_ring.off()

def led_error():
    """Red flash - Error state"""
    pixel_ring.set_color(r=255, g=0, b=0)
```

**Integration Points in `johnny5.py`:**
- `on_open()` → `led_listening()`
- `on_message()` with `UserInput` → `led_thinking()`
- `on_message()` with audio chunks → `led_speaking()`
- After playback complete → `led_listening()`
- `on_error()` → `led_error()`

**Estimated Effort:** 1-2 hours

---

## Priority 2: Direction of Arrival (DOA)

**Why:** Know WHERE sound is coming from. Combined with YOLO, Johnny can look at who's speaking even without seeing their face.

**The ReSpeaker Advantage:** The XMOS XVF-3000 DSP calculates DOA on-chip and exposes it via USB HID.

**Installation:**
```bash
pip install pyusb --break-system-packages
# May need udev rules for non-root access
sudo cp /path/to/respeaker-rules /etc/udev/rules.d/
```

**Implementation - Direct XMOS Access:**
```python
import usb.core
import usb.util

class ReSpeakerDOA:
    """Extract DOA from ReSpeaker XMOS DSP"""

    VENDOR_ID = 0x2886
    PRODUCT_ID = 0x0018

    # XMOS XVF-3000 register addresses
    REGISTER_DOA = 0x21
    REGISTER_VAD = 0x20
    REGISTER_LED = 0x00

    def __init__(self):
        self.dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
        if not self.dev:
            raise RuntimeError("ReSpeaker not found")

    def get_direction(self) -> int:
        """Get direction of arrival in degrees (0-359)"""
        # Read from XMOS register
        result = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, self.REGISTER_DOA, 0, 2
        )
        return int.from_bytes(result, 'little')

    def is_voice_active(self) -> bool:
        """Check if voice activity detected"""
        result = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, self.REGISTER_VAD, 0, 2
        )
        return bool(int.from_bytes(result, 'little'))
```

**Integration with Gantry:**
```python
# Map DOA angle to gantry pan position
def doa_to_gantry_pan(doa_degrees: int) -> int:
    """Convert DOA (0-359) to gantry servo position"""
    # ReSpeaker 0° is forward, gantry center is ~90
    # Adjust based on physical mounting orientation
    offset = 0  # Calibrate this based on mic orientation
    adjusted = (doa_degrees + offset) % 360

    # Map to servo range (e.g., 0-180)
    if adjusted <= 180:
        return 90 - int(adjusted * 90 / 180)
    else:
        return 90 + int((360 - adjusted) * 90 / 180)
```

**Use Cases:**
1. Johnny looks toward speaker even when face is obscured
2. Multiple people: track active speaker
3. "Blindfolded demo" - cap over camera, still follows voice

**Estimated Effort:** 3-4 hours

---

## Priority 3: Speaker Identification with WeSpeaker

**Why:** Recognize WHO is speaking by voice alone. Key for the "blindfolded demo" - identify people without seeing their face.

**The Stack:**
```
Audio → Silero VAD → Segment speech → WeSpeaker ECAPA-TDNN → 192-d embedding → Cosine similarity
```

**Installation:**
```bash
pip install wespeaker silero-vad onnxruntime --break-system-packages

# Download model
wget https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn-LM/resolve/main/model.onnx \
    -O models/wespeaker_ecapa.onnx
```

**Implementation:**
```python
import numpy as np
import onnxruntime as ort
from silero_vad import load_silero_vad, get_speech_timestamps

class SpeakerIdentifier:
    def __init__(self, model_path: str, threshold: float = 0.55):
        self.session = ort.InferenceSession(model_path)
        self.vad = load_silero_vad(onnx=True)
        self.threshold = threshold
        self.enrolled_speakers = {}  # name -> embedding

    def extract_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract 192-d speaker embedding from audio"""
        # Get speech segments
        timestamps = get_speech_timestamps(audio, self.vad, sampling_rate=sr)
        if not timestamps:
            return None

        # Concatenate speech segments
        speech = np.concatenate([audio[t['start']:t['end']] for t in timestamps])

        # Run WeSpeaker
        embedding = self.session.run(None, {'audio': speech.astype(np.float32)})[0]
        return embedding / np.linalg.norm(embedding)  # L2 normalize

    def enroll(self, name: str, audio: np.ndarray):
        """Enroll a speaker with their voice"""
        embedding = self.extract_embedding(audio)
        if embedding is not None:
            self.enrolled_speakers[name] = embedding

    def identify(self, audio: np.ndarray) -> tuple[str, float]:
        """Identify speaker from audio, returns (name, confidence)"""
        embedding = self.extract_embedding(audio)
        if embedding is None:
            return None, 0.0

        best_match, best_score = None, 0.0
        for name, enrolled_emb in self.enrolled_speakers.items():
            score = np.dot(embedding, enrolled_emb)  # Cosine similarity
            if score > best_score:
                best_match, best_score = name, score

        if best_score >= self.threshold:
            return best_match, best_score
        return None, best_score
```

**Integration with Gun.js Identity Store:**
- Store voice embeddings alongside face embeddings
- Cross-modal verification: face says "Alan", voice confirms "Alan"
- Fallback: if face occluded, use voice alone

**Estimated Effort:** 4-6 hours

---

## Priority 4: Audio-Visual Fusion (DOA + YOLO)

**Why:** When multiple people are in frame, know which face is speaking.

**The Fusion Algorithm:**
```
1. YOLO detects faces → bounding boxes with pixel coordinates
2. DOA gives angle in degrees
3. Map DOA angle to camera pixel column
4. Score each face by proximity to DOA line
5. Highest score = active speaker
```

**Implementation:**
```python
import numpy as np
from scipy.stats import norm

class AudioVisualFusion:
    def __init__(self, camera_hfov: float = 72.0, image_width: int = 640):
        self.hfov = camera_hfov  # OAK-D horizontal FOV
        self.width = image_width
        self.doa_sigma = 10.0  # DOA uncertainty ±10°

    def doa_to_pixel(self, doa_degrees: float) -> int:
        """Convert DOA angle to image x-coordinate"""
        # Assuming camera faces forward (0°), DOA 0° = image center
        # Adjust offset based on physical mic/camera alignment
        relative_angle = doa_degrees
        if relative_angle > 180:
            relative_angle -= 360

        # Map angle to pixel
        pixel_x = self.width / 2 + (relative_angle / (self.hfov / 2)) * (self.width / 2)
        return int(np.clip(pixel_x, 0, self.width - 1))

    def score_detections(self, detections: list, doa_degrees: float) -> list:
        """Score YOLO detections by DOA alignment"""
        doa_pixel = self.doa_to_pixel(doa_degrees)
        pixel_sigma = (self.doa_sigma / self.hfov) * self.width

        scored = []
        for det in detections:
            # Get center of bounding box
            cx = (det['x1'] + det['x2']) / 2

            # Gaussian likelihood based on distance from DOA line
            likelihood = norm.pdf(cx, loc=doa_pixel, scale=pixel_sigma)
            scored.append({**det, 'doa_score': likelihood})

        return sorted(scored, key=lambda x: x['doa_score'], reverse=True)

    def get_active_speaker(self, detections: list, doa_degrees: float,
                           vad_active: bool) -> dict:
        """Get the detection most likely to be the active speaker"""
        if not vad_active or not detections:
            return None

        scored = self.score_detections(detections, doa_degrees)
        return scored[0] if scored else None
```

**Estimated Effort:** 2-3 hours (after DOA is working)

---

## Priority 5: Raw Multi-Channel Audio Mode

**Why:** Access individual microphone channels for advanced spatial processing, custom beamforming, or ML training data.

**Firmware Switch Required:**
The ReSpeaker can operate in two modes:
1. **1-channel mode (default):** Processed audio only
2. **6-channel mode:** 4 raw mics + 2 processed channels

**Switching Firmware:**
```bash
# Download 6-channel firmware from Seeed
# Flash via DFU mode (hold button while plugging in)
sudo dfu-util -d 2886:0018 -D respeaker_6mic_firmware.bin
```

**Accessing 6 Channels:**
```python
import sounddevice as sd
import numpy as np

def record_multichannel(duration: float = 5.0, sr: int = 16000) -> np.ndarray:
    """Record 6 channels from ReSpeaker"""
    # Find ReSpeaker device
    devices = sd.query_devices()
    respeaker_idx = None
    for i, d in enumerate(devices):
        if 'ReSpeaker' in d['name'] and d['max_input_channels'] >= 6:
            respeaker_idx = i
            break

    if respeaker_idx is None:
        raise RuntimeError("6-channel ReSpeaker not found")

    # Record all 6 channels
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=6,
                   device=respeaker_idx, dtype='float32')
    sd.wait()
    return audio

# Channel mapping:
# Channel 0: Mic 1 (raw)
# Channel 1: Mic 2 (raw)
# Channel 2: Mic 3 (raw)
# Channel 3: Mic 4 (raw)
# Channel 4: Processed (beamformed)
# Channel 5: Processed (playback reference for AEC)
```

**Use Cases:**
- Custom SRP-PHAT DOA (using pyroomacoustics)
- Training data for neural beamforming
- Spatial audio recording for debugging

**Estimated Effort:** 2-3 hours (mostly firmware flashing)

---

## Implementation Order Recommendation

```
Phase 1: Visual Polish (Day 1)
├── LED Ring Feedback
└── Integration with johnny5.py events

Phase 2: Spatial Awareness (Day 1-2)
├── DOA from XMOS DSP
├── Gantry tracking to DOA
└── "Look at speaker" behavior

Phase 3: Voice Identity (Day 2-3)
├── WeSpeaker model download + TensorRT optimization
├── Silero VAD integration
├── Speaker enrollment flow
└── Gun.js voice embedding storage

Phase 4: Multimodal Fusion (Day 3)
├── Audio-Visual Fusion module
├── DOA + YOLO integration
└── Confidence scoring for identity

Phase 5: Advanced (Stretch Goals)
├── 6-channel raw audio mode
├── Custom neural beamforming
└── Speaker diarization during conversations
```

---

## Dependencies to Install

```bash
# LED Control
pip install pixel_ring --break-system-packages

# USB Access for DOA
pip install pyusb --break-system-packages
sudo usermod -a -G plugdev $USER  # USB permissions

# Speaker Identification
pip install wespeaker silero-vad onnxruntime --break-system-packages

# Spatial Audio (optional, for custom DOA)
pip install pyroomacoustics --break-system-packages

# Download WeSpeaker model
mkdir -p models
wget https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn-LM/resolve/main/model.onnx \
    -O models/wespeaker_ecapa.onnx
```

---

## Hardware Considerations

### Microphone Orientation
- Document the physical mounting orientation of ReSpeaker
- DOA 0° direction relative to robot front
- Offset calibration value for `doa_to_gantry_pan()`

### USB Bandwidth
- ReSpeaker + OAK-D both on USB
- Consider USB hub with dedicated bandwidth
- 6-channel mode uses more bandwidth than 1-channel

### LED Brightness
- 12 LEDs at full brightness = noticeable power draw
- Consider dimming for battery life
- LEDs visible in camera frame? May affect face detection

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/led_controller.py` | LED ring state machine |
| `src/respeaker_doa.py` | DOA extraction from XMOS |
| `src/speaker_id.py` | WeSpeaker voice identification |
| `src/fusion.py` | Audio-visual fusion scoring |
| `src/gantry_tracker.py` | DOA → gantry servo mapping |

---

## Success Metrics

1. **LED Feedback:** Users can tell Johnny's state at a glance
2. **DOA Accuracy:** ±15° or better in typical room
3. **Speaker ID:** >90% accuracy with 10s enrollment
4. **Fusion:** Correctly identifies active speaker in 2+ person scene
5. **Blindfolded Demo:** Identify and greet by voice alone

---

## Open Questions for Discussion

1. **LED Priority vs. DOA:** Quick visual win or core spatial feature first?
2. **Speaker Enrollment UX:** How do users enroll their voice? On first meeting? Explicit command?
3. **Firmware Mode:** Stay with 1-channel or switch to 6-channel? Trade-offs?
4. **DOA + Gantry Latency:** How fast should Johnny turn to face speaker?
5. **Conflict Resolution:** What if face says "Alan" but voice says "Bob"?
