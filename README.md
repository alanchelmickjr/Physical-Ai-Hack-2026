# Physical AI Hack 2026: Johnny Five

> **"Number 5 is alive!"** — A social robot that connects with and remembers people.

![Hackathon](https://img.shields.io/badge/Physical%20AI-Hack%202026-blue)
![Status](https://img.shields.io/badge/Status-Functional-green)
![Platform](https://img.shields.io/badge/Jetson-Orin%208GB-orange)
![Servos](https://img.shields.io/badge/Servos-19-purple)

---

## What is Johnny Five?

**Johnny Five** is a robot MODEL (like "iPhone"). Each unit picks its own name — this one chose **Chloe**.

Chloe is designed for **human connection**, not task automation. She:
- Remembers everyone she meets (face + voice embeddings)
- Tracks who is speaking and looks at them
- Engages in natural emotional conversation
- Uses gestures and body language to express herself
- Detects safety hazards (fire, cords) and alerts humans

**Key demo:** Cover the camera with a cap — Chloe still knows who's talking by voice alone.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  HUME EVI (Conscious Mind)                              │
│  Conversation, personality, decisions                   │
│  "Hello Alan!" → greets by name                         │
└────────────────────────┬────────────────────────────────┘
                         │ events
┌────────────────────────▼────────────────────────────────┐
│  SPINE (Autonomic / Muscle Memory)                      │
│  Reflexes that run automatically:                       │
│  • Head tracking (look at speaker)                      │
│  • Gestures while talking                               │
│  • Fire/smoke detection → alert                         │
│  • Terrain navigation (gaps, cords, rails)              │
└────────────────────────┬────────────────────────────────┘
                         │ adapter
┌────────────────────────▼────────────────────────────────┐
│  HARDWARE (19 Servos via Solo-CLI)                      │
│  ACM0: Left arm, wheels, lift, hitch                    │
│  ACM1: Right arm, gantry                                │
└─────────────────────────────────────────────────────────┘
```

See [docs/AUTONOMIC_ARCHITECTURE.md](docs/AUTONOMIC_ARCHITECTURE.md) for detailed diagrams.

---

## Hardware

| Component | Spec | Purpose |
|-----------|------|---------|
| **Brain** | Jetson Orin 8GB | Edge AI inference |
| **Camera** | OAK-D on 2-DOF gantry | Face detection + depth + head tracking |
| **Microphone** | ReSpeaker 4-Mic USB Array | Voice ID + DOA (direction of arrival) |
| **Arms** | Dual SO101 (6-DOF each) | Gestures & manipulation |
| **Base** | 3-wheel Mecanum | Omnidirectional movement |
| **Lift** | 30cm vertical | Height adjustment |
| **Hitch** | Rear grabber | IKEA cart towing + charger docking |
| **Power** | Anker Solix 12V | Portable operation |

### Motor Layout (19 Servos)

| Bus | Subsystem | Motor IDs | Count |
|-----|-----------|-----------|-------|
| ACM0 | Left Arm | 1-6 | 6 |
| ACM0 | Wheels | 7-9 | 3 |
| ACM0 | Lift | 10 | 1 |
| ACM0 | Hitch | 11 | 1 |
| ACM1 | Right Arm | 1-6 | 6 |
| ACM1 | Gantry | 7-8 | 2 |

All servos: Dynamixel XL330-M288-T, Protocol 2.0

---

## Features

### Multimodal Identity
| Modality | Technology | Status |
|----------|------------|--------|
| **Face** | WhoAmI + YOLO + DeepFace | Working |
| **Voice Conversation** | Hume EVI + ReSpeaker | Working |
| **Voice ID** | WeSpeaker ECAPA-TDNN (192-d) | Working |
| **Direction of Arrival** | SRP-PHAT 4-mic fusion | Working |

### Autonomic Behaviors (Always Running)
- **Head Tracking** — Gantry follows speaker via DOA
- **Expressive Gestures** — Arms move naturally while speaking
- **Polite Pointing** — Points at person when first mentioning name (rate-limited, not rude)
- **Fire Detection** — Both arms point + twitch at fire (visible even with headphones)
- **Terrain Navigation** — Detects elevator gaps, door rails, cords

### Safety Features
- **Fire/Smoke Detection** — Visual detection via OAK-D
- **Cord Avoidance** — Detects and goes around floor cables
- **Wheel Drag Detection** — Emergency stop if cord caught
- **Emergency Stop** — Direct torque disable on all motors

---

## Project Structure

```
Physical-Ai-Hack-2026/
├── README.md                    # You are here
├── CLAUDE.md                    # AI assistant context
│
├── johnny5.py                   # Hume EVI voice conversation (main entry)
├── motion_coordinator.py        # Spine - autonomic movement control
├── head_tracker.py              # DOA → gantry head tracking
├── doa_reader.py                # ReSpeaker direction of arrival
├── terrain_navigation.py        # Gap/cord/rail detection & crossing
├── visual_safety.py             # Fire/smoke detection
├── led_controller.py            # ReSpeaker LED feedback
├── johnny5_body.py              # Body movement abstraction
├── whoami_full.py               # Face recognition with temporal smoothing
│
├── adapters/                    # Hardware abstraction layer
│   ├── base.py                  # Abstract RobotAdapter interface
│   └── johnny5.py               # Solo-CLI implementation for Chloe
│
├── tools/                       # Hume EVI tool system
│   ├── registry.py              # 50+ tool definitions
│   ├── engine.py                # Tool execution with dependencies
│   ├── realtime.py              # 30Hz command queue
│   └── verbal_calibration.py    # Human-assisted motor setup
│
├── docs/                        # Architecture & planning
│   ├── AUTONOMIC_ARCHITECTURE.md    # Hume ↔ Spine ↔ Adapter diagrams
│   ├── JOHNNY5_HARDWARE_SPEC.md     # 19-servo layout & calibration
│   ├── HUME_EVI_ROBOT_CONTROL_PLAN.md
│   └── ...
│
└── forks/                       # Modified dependencies
    └── chloe-lerobot/           # LeRobot fork with Chloe config
```

---

## Quick Start

### Run the Full Demo

```bash
# SSH to Jetson
ssh robbie@192.168.88.44  # password: robot

# Start everything
./start_johnny5.sh
```

Johnny 5 will:
1. See you through the camera
2. Recognize your face
3. Track you with the gantry as you speak
4. Greet you by name with emotional voice
5. Gesture naturally during conversation

### Run Components Separately

```bash
# Face recognition only
DISPLAY=:0 python3 whoami_full.py

# Voice conversation only
python3 johnny5.py

# Test head tracking
python3 head_tracker.py

# Test motor control
solo robo --port /dev/ttyACM0 --ids 1,2,3,4,5,6 --status
```

### Calibrate Motors

```bash
# Interactive calibration with voice guidance
python3 tools/verbal_calibration.py

# Or via Solo-CLI
solo robo --calibrate all
```

---

## Platform Portability

The adapter pattern allows the same spine to run on different robots:

```python
# Johnny5 (this robot)
adapter = Johnny5Adapter()  # Solo-CLI → Dynamixel

# Future: OpenDroids
adapter = OpenDroidAdapter()  # Different motors, same spine

# The spine queries capabilities and adapts
caps = adapter.get_capabilities()
# → {arms: [left, right], base_type: "mecanum", ...}
```

See [docs/AUTONOMIC_ARCHITECTURE.md](docs/AUTONOMIC_ARCHITECTURE.md) for details.

---

## Key Files

| File | Purpose |
|------|---------|
| `johnny5.py` | Main entry - Hume EVI WebSocket + face recognition triggers |
| `motion_coordinator.py` | The "spine" - coordinates all movement, gestures, safety |
| `adapters/johnny5.py` | Hardware adapter - translates intents to Solo-CLI commands |
| `tools/registry.py` | 50+ tools Hume can call (wave, point, move, etc.) |
| `terrain_navigation.py` | Autonomous obstacle handling (gaps, rails, cords) |
| `visual_safety.py` | Fire/smoke detection with auto-alert |

---

## Documentation

- [AUTONOMIC_ARCHITECTURE.md](docs/AUTONOMIC_ARCHITECTURE.md) — Hume ↔ Spine ↔ Adapter with mermaid diagrams
- [JOHNNY5_HARDWARE_SPEC.md](docs/JOHNNY5_HARDWARE_SPEC.md) — 19-servo layout, calibration, motor IDs
- [HUME_EVI_ROBOT_CONTROL_PLAN.md](docs/HUME_EVI_ROBOT_CONTROL_PLAN.md) — Tool execution architecture
- [JOHNNY5_FLOW_ANALYSIS.md](docs/JOHNNY5_FLOW_ANALYSIS.md) — Voice pipeline debugging

---

## References

- [Hume AI EVI](https://hume.ai) — Emotional voice interface
- [WhoAmI](https://github.com/alanchelmickjr/whoami) — Face recognition for OAK-D
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker) — Speaker verification (192-d embeddings)
- [Solo-CLI](https://github.com/huggingface/lerobot) — Dynamixel motor control
- [ReSpeaker](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/) — 4-mic USB array
- [face_recognition](https://github.com/ageitgey/face_recognition) — dlib-based face embeddings

---

## License

MIT — Hack away!
