# Johnny Five — Multimodal Identity System
## Claude Code Implementation Guide

**One document. Everything you need.**

---

## Overview

Johnny Five identifies people through **three modalities**:
1. **Face** — WhoAmI (DeepFace on OAK-D) ✅ Already working
2. **Voice** — WeSpeaker ECAPA-TDNN speaker embeddings
3. **Direction** — 4-mic DOA fused with YOLO to know WHO is speaking

Demo: Put a cap over OAK-D. Johnny still knows who's talking by voice alone.

---

## Hardware

| Component | Spec |
|-----------|------|
| Brain | Jetson Orin 8GB |
| Camera | OAK-D on 2-DOF gantry |
| Mic | 4-way directional USB (ReSpeaker or similar) |
| Arms | Dual SO101 via Solo-CLI |
| Power | Anker Solix 12V direct |

**Bus Layout:**
- ACM0: Left arm (1-6), lift (10), wheels (7-9)
- ACM1: Right arm (1-6), gantry (7-8)

---

## Dependencies

```bash
# Core
pip install wespeaker silero-vad pyroomacoustics --break-system-packages
pip install depthai opencv-python ultralytics --break-system-packages

# Solo-CLI for motor control
pip install solo-cli --break-system-packages

# Gun.js (Node side)
npm install gun
```

---

## 1. Voice Speaker ID

### Model: WeSpeaker ECAPA-TDNN (192-d embeddings)

```python
# speaker_id.py
import numpy as np
import onnxruntime as ort
from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf

class SpeakerIdentifier:
    def __init__(self, model_path="wespeaker_ecapa.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.vad = load_silero_vad(onnx=True)
        self.enrolled = {}  # name -> embedding
        
    def extract_embedding(self, audio_16k: np.ndarray) -> np.ndarray:
        """Extract 192-d speaker embedding."""
        # Compute mel spectrogram (80 bins)
        mel = self._compute_mel(audio_16k)
        mel = mel[np.newaxis, :, :].astype(np.float32)
        
        embedding = self.session.run(None, {"audio": mel})[0][0]
        return embedding / np.linalg.norm(embedding)
    
    def enroll(self, name: str, audio_samples: list[np.ndarray]):
        """Enroll speaker from multiple audio clips."""
        embeddings = [self.extract_embedding(a) for a in audio_samples]
        centroid = np.mean(embeddings, axis=0)
        self.enrolled[name] = centroid / np.linalg.norm(centroid)
        
    def identify(self, audio_16k: np.ndarray, threshold=0.55) -> tuple[str, float]:
        """Identify speaker. Returns (name, confidence) or (None, 0)."""
        # Check VAD first
        if not get_speech_timestamps(audio_16k, self.vad, sampling_rate=16000):
            return None, 0.0
            
        emb = self.extract_embedding(audio_16k)
        
        best_name, best_score = None, 0.0
        for name, enrolled_emb in self.enrolled.items():
            score = float(np.dot(emb, enrolled_emb))
            if score > best_score:
                best_name, best_score = name, score
                
        if best_score >= threshold:
            return best_name, best_score
        return None, best_score
    
    def _compute_mel(self, audio, sr=16000, n_mels=80):
        """Compute log mel spectrogram."""
        import librosa
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        return librosa.power_to_db(mel, ref=np.max).T
```

### TensorRT optimization (optional, 4x faster):
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=wespeaker_ecapa.onnx \
    --saveEngine=speaker.trt \
    --fp16
```

---

## 2. Audio-Visual Fusion (DOA + YOLO)

### Direction of Arrival from 4-mic array

```python
# doa_tracker.py
import numpy as np

try:
    # ReSpeaker built-in DOA
    import usb.core
    from tuning import Tuning
    RESPEAKER = True
except ImportError:
    RESPEAKER = False

class DOATracker:
    def __init__(self):
        if RESPEAKER:
            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            self.mic = Tuning(dev) if dev else None
        else:
            self.mic = None
            
    def get_direction(self) -> float:
        """Get sound direction in degrees (-180 to 180, 0 = forward)."""
        if self.mic:
            angle = self.mic.direction  # 0-359
            return angle - 360 if angle > 180 else angle
        return 0.0
    
    def is_speaking(self) -> bool:
        """Check if voice activity detected."""
        return self.mic.is_voice() if self.mic else False
```

### Fuse DOA with YOLO detections

```python
# fusion.py
import numpy as np

def doa_to_pixel_x(doa_deg, hfov=68.8, width=640):
    """Convert DOA angle to image x-coordinate."""
    cx = width / 2
    f = width / (2 * np.tan(np.radians(hfov / 2)))
    return cx + f * np.tan(np.radians(doa_deg))

def identify_speaker(doa_deg, detections, doa_sigma=10.0):
    """
    Score YOLO detections by proximity to DOA.
    
    Args:
        doa_deg: Sound direction from mic array
        detections: List of (bbox, name, face_confidence)
        doa_sigma: DOA uncertainty in degrees
    
    Returns:
        Best matching detection or None
    """
    if not detections:
        return None
        
    doa_x = doa_to_pixel_x(doa_deg)
    
    scored = []
    for bbox, name, conf in detections:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        # Pixel distance to DOA line
        dist = abs(center_x - doa_x)
        
        # Gaussian likelihood
        likelihood = np.exp(-0.5 * (dist / (doa_sigma * 5))**2)
        score = likelihood * conf
        
        scored.append((score, name, bbox))
    
    scored.sort(reverse=True)
    return scored[0] if scored[0][0] > 0.3 else None
```

---

## 3. Gun.js Identity Storage

```javascript
// identity_store.js
const Gun = require('gun');
const gun = Gun(['http://localhost:8765/gun']);

// Schema: face + voice embeddings together
const people = gun.get('johnny5_people');

function savePerson(id, data) {
    // data = { name, faceEmbedding, voiceEmbedding, lastSeen }
    const faceB64 = float32ToBase64(data.faceEmbedding);
    const voiceB64 = float32ToBase64(data.voiceEmbedding);
    
    people.get(id).put({
        name: data.name,
        face: faceB64,
        voice: voiceB64,
        lastSeen: Date.now()
    });
}

function float32ToBase64(arr) {
    const bytes = new Uint8Array(arr.buffer);
    return btoa(String.fromCharCode(...bytes));
}

function base64ToFloat32(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
}

module.exports = { savePerson, people, base64ToFloat32 };
```

---

## 4. Solo-CLI Motor Control

```bash
# Initial setup
solo robo --motors all
solo robo --calibrate all

# Test teleop
solo robo --teleop

# Record gesture demonstrations
solo robo --record

# Train ACT policy for gestures
solo robo --train

# Run trained policy
solo robo --inference
```

### Programmatic control:
```python
# Use LeRobot directly for custom gestures
from lerobot.common.robot_devices.robots.factory import make_robot

robot = make_robot("alohamini")
robot.connect()

# Wave gesture
robot.send_action({"right_shoulder_pan": 30, "right_elbow": 90, ...})
```

---

## 5. Integration: Main Loop

```python
# main.py
import asyncio
import numpy as np
from speaker_id import SpeakerIdentifier
from doa_tracker import DOATracker
from fusion import identify_speaker

# Assumes WhoAmI is running and provides face detections
from whoami import WhoAmI

class JohnnyFiveIdentity:
    def __init__(self):
        self.speaker_id = SpeakerIdentifier("wespeaker_ecapa.onnx")
        self.doa = DOATracker()
        self.whoami = WhoAmI()
        
    async def identify_current_speaker(self, audio_chunk, frame):
        """
        Multimodal identification:
        1. Face detection (WhoAmI)
        2. DOA (mic array)
        3. Voice ID (WeSpeaker)
        """
        # Get all detected faces
        faces = self.whoami.detect_faces(frame)
        detections = [(f.bbox, f.name, f.confidence) for f in faces]
        
        # Check if someone is speaking
        if not self.doa.is_speaking():
            return None
            
        # Get sound direction
        doa_angle = self.doa.get_direction()
        
        # Find which face matches DOA
        speaker = identify_speaker(doa_angle, detections)
        
        if speaker:
            score, face_name, bbox = speaker
            
            # Verify with voice
            voice_name, voice_conf = self.speaker_id.identify(audio_chunk)
            
            # Multimodal fusion
            if face_name and voice_name:
                if face_name == voice_name:
                    return {"name": face_name, "confidence": "high", "modalities": ["face", "voice", "doa"]}
                else:
                    # Conflict - trust face if it's strong
                    return {"name": face_name, "confidence": "medium", "note": "voice_mismatch"}
            elif voice_name:
                return {"name": voice_name, "confidence": "medium", "modalities": ["voice", "doa"]}
            elif face_name:
                return {"name": face_name, "confidence": "medium", "modalities": ["face", "doa"]}
        
        # Voice only (cap over camera demo)
        voice_name, voice_conf = self.speaker_id.identify(audio_chunk)
        if voice_name:
            return {"name": voice_name, "confidence": "voice_only", "modalities": ["voice"]}
            
        return None
    
    async def enroll_during_conversation(self, name, face_embedding, audio_clips):
        """Enroll new person with both face and voice."""
        # Face is handled by WhoAmI
        self.whoami.enroll_face(name, face_embedding)
        
        # Voice enrollment
        self.speaker_id.enroll(name, audio_clips)
        
        print(f"Enrolled {name} with face + voice")
```

---

## 6. Demo Script: Blindfolded Johnny

```python
# demo_blindfolded.py
"""
Put a cap over OAK-D camera.
Johnny still knows who's talking by voice alone.
"""

import asyncio
from main import JohnnyFiveIdentity

async def blindfolded_demo():
    johnny = JohnnyFiveIdentity()
    
    print("Johnny Five is blindfolded!")
    print("Speak to him - he knows who you are by voice.")
    
    while True:
        # Get audio chunk from mic
        audio = await get_audio_chunk()  # Your audio capture
        
        # Voice-only identification
        name, confidence = johnny.speaker_id.identify(audio)
        
        if name:
            print(f"I hear you, {name}! (confidence: {confidence:.2f})")
            # Johnny responds even without seeing
            await johnny.speak(f"Hello {name}! I recognize your voice!")
        
        await asyncio.sleep(0.1)
```

---

## File Structure

```
johnny5/
├── config/
│   ├── motors.yaml
│   └── personality.yaml
├── models/
│   ├── wespeaker_ecapa.onnx      # Download from HuggingFace
│   └── speaker.trt               # TensorRT converted (optional)
├── src/
│   ├── main.py                   # Integration loop
│   ├── speaker_id.py             # Voice identification
│   ├── doa_tracker.py            # Mic array DOA
│   ├── fusion.py                 # Audio-visual fusion
│   └── identity_store.js         # Gun.js storage
├── requirements.txt
└── README.md
```

---

## Quick Test Commands

```bash
# Test speaker ID
python -c "
from speaker_id import SpeakerIdentifier
sid = SpeakerIdentifier('models/wespeaker_ecapa.onnx')
print('Speaker ID loaded')
"

# Test DOA (requires ReSpeaker)
python -c "
from doa_tracker import DOATracker
doa = DOATracker()
print(f'DOA: {doa.get_direction()}°, Speaking: {doa.is_speaking()}')
"

# Test Solo-CLI motors
solo robo --motors all
```

---

## Model Downloads

```bash
# WeSpeaker ECAPA-TDNN
wget https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn-LM/resolve/main/model.onnx \
    -O models/wespeaker_ecapa.onnx

# Silero VAD (auto-downloads on first use)
python -c "from silero_vad import load_silero_vad; load_silero_vad(onnx=True)"
```

---

## Summary

| Modality | Model | Embedding | Latency |
|----------|-------|-----------|---------|
| Face | DeepFace (WhoAmI) | 512-d | ~30ms |
| Voice | WeSpeaker ECAPA | 192-d | ~15ms TRT |
| Direction | ReSpeaker DOA | angle | ~10ms |

**Fusion logic:**
1. If face + voice + DOA agree → high confidence
2. If face + DOA agree → medium confidence  
3. If voice only (blindfolded) → voice confidence
4. Conflicts → trust face, log for review

**The demo that wins:** Cap over camera, Johnny still greets everyone by name.
