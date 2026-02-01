# Physical AI Hack 2026: Johnny Five

> **"Number 5 is alive!"** — A multimodal identity robot that knows who you are.

![Hackathon](https://img.shields.io/badge/Physical%20AI-Hack%202026-blue)
![Status](https://img.shields.io/badge/Face%20Recognition-Working-green)
![Platform](https://img.shields.io/badge/Jetson-Orin%208GB-orange)

---

## What is Johnny Five?

Johnny Five is a robot that identifies people through **multiple modalities**:

| Modality | Technology | Status |
|----------|------------|--------|
| **Face** | WhoAmI + YOLO + face_recognition | ✅ Working |
| **Voice Conversation** | Hume EVI + ReSpeaker 4-mic | ✅ Working |
| **Voice ID** | WeSpeaker ECAPA-TDNN | 🚧 Next |
| **Direction** | 4-mic DOA array | 🚧 Planned |

**The demo:** Johnny 5 sees you, greets you by name, and talks with you using emotional voice.

---

## Hardware Stack

| Component | Spec | Purpose |
|-----------|------|---------|
| **Brain** | Jetson Orin 8GB | Edge AI inference |
| **Camera** | OAK-D S3 | Face detection + depth |
| **Display** | 7" Touchscreen (800x480) | Fullscreen UI |
| **Arms** | Dual SO101 (AlohaMini) | Gestures & interaction |
| **Mic** | 4-way USB array | Voice ID + DOA |
| **Power** | Anker Solix 12V | Portable operation |

---

## Current Status

### ✅ Face Recognition (WhoAmI)

Face recognition is **live and working** on the Jetson Orin:

- **8 enrolled faces**: AL, Jordan, Sam, Jack, Vitaly, Armin, Emerson, Tigran
- **YOLO v8** for person detection
- **face_recognition** library (128-d embeddings)
- **Temporal smoothing** — names lock after 2 positive IDs, no flickering
- **Fullscreen display** on 7" touchscreen

```
Green box = Known person (name displayed)
Yellow box = Unidentified
* prefix = Identity locked (high confidence)
```

### ✅ Voice Conversation (Hume EVI)

Emotional voice conversation is **live and working**:

- **Hume AI EVI** for emotional, natural voice responses
- **ReSpeaker 4-Mic Array** for voice input
- **Johnny 5 Persona** — speaks with curiosity and enthusiasm
- **Face → Voice Integration** — greets people by name when recognized
- **HDMI Audio Output** for speaker playback
- **Acoustic Echo Prevention** — mic auto-mutes during speech to prevent feedback loops
- **Latency Instrumentation** — millisecond timing for debugging

```
[06:02:39.098] Connected! Number 5 is alive! (connect took 581ms)
[06:02:45.123] FACE DETECTED: Hello Jordan!
[06:02:45.789] FIRST AUDIO CHUNK (96044 bytes) - latency: 664ms
[06:02:45.789] MIC MUTED (audio playing)
[06:02:48.043] AUDIO COMPLETE: 4 chunks, 327974 bytes, 2254ms playback
[06:02:48.343] MIC UNMUTED (audio done)
```

### 🚧 Voice Recognition (Next)

WeSpeaker ECAPA-TDNN for speaker identification:
- 192-d voice embeddings
- Works even when camera is covered
- Silero VAD for speech detection

### 🚧 Direction of Arrival (Planned)

4-mic array fusion with YOLO:
- Know WHO is speaking, not just detect speech
- Fuse audio direction with face bounding boxes

---

## Project Structure

```
Physical-Ai-Hack-2026/
├── README.md              # You are here
├── CLAUDE.md              # AI assistant context
├── johnny5.py             # Hume EVI voice conversation (main)
├── whoami_full.py         # Face recognition with temporal smoothing
├── start_johnny5.sh       # Launches both services
├── enroll_face.py         # Add new faces to database
├── docs/                  # Planning & architecture
│   ├── JOHNNY5_FLOW_ANALYSIS.md   # Voice flow diagrams & debugging
│   ├── JOHNNY5_MULTIMODAL_IDENTITY.md
│   └── ...
├── whoami/                # Face recognition system (submodule)
├── AlohaMini/             # Robot arm control (submodule)
├── XLeRobot/              # Motor/teleoperation (submodule)
└── lets-connect/          # Networking utilities
```

---

## Quick Start

### Run the Full Demo

Double-click the **Johnny 5 Demo** icon on the Jetson desktop, or:

```bash
ssh robbie@192.168.88.44  # password: robot
cd /home/robbie
./start_johnny5.sh
```

This starts both face recognition AND voice conversation. Johnny 5 will:
1. See you through the camera
2. Recognize your face
3. Greet you by name with emotional voice
4. Chat with you using Hume AI's EVI

### Run Components Separately

**Face Recognition Only:**
```bash
DISPLAY=:0 python3 whoami_full.py
```

**Voice Only:**
```bash
python3 johnny5.py
```

### Enroll a New Face

```bash
python3 enroll_face.py "YourName"
# Press SPACE to capture, Q to quit
```

---

## The Tech

### Face Recognition Pipeline

```
OAK-D Camera
    ↓
YOLO v8 (person detection, class 0)
    ↓
Extract face region (upper 40% of bbox)
    ↓
face_recognition (128-d embeddings)
    ↓
Compare to enrolled faces (threshold < 0.55)
    ↓
Temporal smoothing (ignore "Unidentified" votes)
    ↓
Lock identity after 2 positive matches
    ↓
Display with colored bounding box
```

### Voice Conversation Pipeline

```
Face Recognition (whoami_full.py)
    ↓
Writes greeting to /tmp/johnny5_greeting.txt
    ↓
johnny5.py polls file every 0.5s
    ↓
Sends to Hume EVI via WebSocket
    ↓
Hume processes with emotional AI
    ↓
Audio chunks stream back (base64)
    ↓
MIC MUTED → Audio plays via HDMI → MIC UNMUTED
```

See [docs/JOHNNY5_FLOW_ANALYSIS.md](docs/JOHNNY5_FLOW_ANALYSIS.md) for detailed architecture diagrams.

### Key Innovation: Smart Smoothing

Previous approaches flickered between names. Our solution:

```python
class TrackedPerson:
    def vote(self, name, conf):
        # Only count REAL names, not Unidentified
        if name != "Unidentified":
            self.name_votes[name] += 1
            if self.name_votes[best_name] >= 2:
                self.locked_name = best_name  # Lock it!
```

This means:
- ✅ Positive IDs accumulate
- ✅ "Unidentified" frames are ignored
- ✅ Identity locks after consistent recognition
- ✅ No more AL → Unknown → AL flickering

---

## Team

Built overnight for Physical AI Hack 2026.

---

## References

- [Hume AI EVI](https://hume.ai) — Emotional voice interface
- [WhoAmI](https://github.com/alanchelmickjr/whoami) — Face recognition for OAK-D
- [AlohaMini](https://github.com/alanch/alohamini) — Dual arm robot control
- [XLeRobot](https://github.com/alanch/xlerobot) — Teleoperation framework
- [face_recognition](https://github.com/ageitgey/face_recognition) — dlib-based face embeddings
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker) — Speaker verification
- [ReSpeaker](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/) — 4-mic USB array

---

## License

MIT — Hack away!
