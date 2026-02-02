# Physical AI Hack 2026: Johnny Five

> **"Number 5 is alive!"** — A social robot that connects with and remembers people.

![Hackathon](https://img.shields.io/badge/Physical%20AI-Hack%202026-blue)
![Status](https://img.shields.io/badge/Status-Functional-green)
![Portable](https://img.shields.io/badge/Multi--Robot-Portable-orange)
[![Hume EVI](https://img.shields.io/badge/Voice-Hume_EVI-FF6B6B)](https://hume.ai)

**Johnny Five** · ![Orin 8GB](https://img.shields.io/badge/Orin-8GB-76B900) [![OAK-D](https://img.shields.io/badge/OAK--D_Pro-00A0DC)](https://docs.luxonis.com/) [![ReSpeaker](https://img.shields.io/badge/ReSpeaker_4--Mic-00C853)](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) [![Solo-CLI](https://img.shields.io/badge/Solo--CLI-FF9800)](https://github.com/TheRobotStudio/SO-ARM100)

**Booster K1** · ![Orin NX](https://img.shields.io/badge/Orin-NX-76B900) [![ZED X](https://img.shields.io/badge/ZED_X-00A0DC)](https://www.stereolabs.com/) ![6-Mic](https://img.shields.io/badge/6--Mic_Array-00C853) ![ROS2](https://img.shields.io/badge/ROS2-22314E)

**Unitree G1** · ![Orin NX](https://img.shields.io/badge/Orin-NX-76B900) [![RealSense](https://img.shields.io/badge/RealSense_D435-0071C5)](https://www.intelrealsense.com/) ![4-Mic](https://img.shields.io/badge/4--Mic_Array-00C853) ![unitree_sdk2](https://img.shields.io/badge/unitree__sdk2-333)

### Related Repositories

| Repo | Purpose |
|------|---------|
| [memoRable](https://github.com/alanchelmickjr/memoRable) | Context-aware memory for AI agents & robots |
| [whoami](https://github.com/alanchelmickjr/whoami) | Secure facial recognition for robots |
| [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Solo-CLI for Feetech servo control |
| [lerobot](https://github.com/huggingface/lerobot) | HuggingFace robotics ML toolkit |

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

### Sensor Fusion (DOA + Depth)

Active speaker identification using multimodal fusion:

```
ReSpeaker 4-Mic Array          OAK-D Pro Stereo
        │                            │
   DOA (0-360°)              Depth + RGB → YOLO
        │                            │
        └──────────┬─────────────────┘
                   │
         DOA-Spatial Fusion
                   │
    ┌──────────────┴──────────────┐
    │  • Angular match (±15° σ)   │
    │  • Depth confidence         │
    │  • Active speaker ID        │
    │  • 3D position (x,y,z mm)   │
    └─────────────────────────────┘
```

| Component | Utilization | Notes |
|-----------|-------------|-------|
| ReSpeaker | ~70% | DOA, VAD, LEDs (AEC not yet connected) |
| OAK-D Pro | ~70% | RGB + stereo depth + visual safety |
| Fusion | ~80% | DOA + depth person matching |

**Usage:**
```python
from doa_spatial_fusion import get_fusion

fusion = get_fusion()
fusion.start()
fusion.enable_visual_safety(enabled=True)

speaker = fusion.get_active_speaker()
if speaker and speaker.is_valid:
    print(f"Speaker at {speaker.detection.distance:.0f}mm")
```

---

## Project Structure

```
Physical-Ai-Hack-2026/
├── README.md                    # You are here
├── CLAUDE.md                    # AI assistant context
├── robot_factory.py             # Unified robot creation entry point
│
├── johnny5.py                   # Hume EVI voice conversation (main entry)
├── motion_coordinator.py        # Spine - autonomic movement control
├── head_tracker.py              # DOA → gantry head tracking
├── doa_reader.py                # ReSpeaker direction of arrival
├── doa_spatial_fusion.py        # DOA + OAK-D depth fusion for active speaker ID
├── spatial_tracker.py           # OAK-D stereo depth person tracking (3D positions)
├── terrain_navigation.py        # Gap/cord/rail detection & crossing
├── visual_safety.py             # Fire/smoke detection
├── led_controller.py            # ReSpeaker LED feedback
├── johnny5_body.py              # Body movement abstraction
├── whoami_full.py               # Face recognition with temporal smoothing
│
├── adapters/                    # Robot hardware adapters (drop-in)
│   ├── base.py                  # Abstract RobotAdapter interface
│   ├── johnny5.py               # Johnny Five (Solo-CLI/Feetech)
│   ├── booster_k1.py            # Booster K1 (ROS2/bipedal)
│   └── unitree_g1.py            # Unitree G1 (unitree_sdk2/ROS2)
│
├── cameras/                     # Camera abstraction layer
│   ├── base.py                  # Abstract SpatialCamera interface
│   ├── oakd.py                  # OAK-D / OAK-D Pro (VPU YOLO)
│   ├── zed.py                   # ZED / ZED X / ZED 2
│   ├── realsense.py             # Intel RealSense D435/D455
│   └── factory.py               # Camera factory
│
├── microphones/                 # Microphone array abstraction
│   ├── base.py                  # Abstract MicrophoneArray interface
│   ├── respeaker.py             # ReSpeaker 4/6-Mic USB arrays
│   ├── circular6.py             # Generic circular 6-mic (SRP-PHAT)
│   ├── unitree.py               # Unitree G1 built-in 4-mic
│   └── factory.py               # Microphone factory
│
├── actions/                     # Unified action primitives
│   ├── base.py                  # Abstract action definitions
│   └── unified.py               # UnifiedActions interface
│
├── config/                      # Configuration
│   ├── hardware.py              # Johnny Five motor config
│   ├── motors.py                # Motor interface
│   └── robots.py                # Multi-robot registry
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

Johnny Five is designed to exist in **multiple bodies simultaneously**. The same mind, memories, and personality can run on different hardware platforms through abstraction layers.

### Supported Platforms

| Platform | Camera | Microphone | Locomotion | Status |
|----------|--------|------------|------------|--------|
| **Johnny Five** | OAK-D Pro | ReSpeaker 4-Mic | Mecanum wheels | Production |
| **Booster K1** | ZED X | Circular 6-Mic | Bipedal (22 DOF) | Supported |
| **Unitree G1** | RealSense D435 | Built-in 4-Mic | Bipedal (23-43 DOF) | Supported |

### Quick Start

```python
from robot_factory import create_robot, RobotType

# Auto-detect hardware
robot = create_robot()

# Or specify platform
robot = create_robot(RobotType.BOOSTER_K1)

# Same API regardless of body
await robot.adapter.connect()
robot.start_sensors()

# Identity persists across bodies via Gun.js
```

### Unified Actions API

```python
from robot_factory import create_robot
from actions import UnifiedActions, Target, Hand, GestureType

robot = create_robot()
await robot.adapter.connect()

actions = UnifiedActions(robot)

# These work on any robot body
await actions.wave()                              # Wave hello
await actions.wave(hand=Hand.LEFT)                # Wave with left hand
await actions.look_at(Target.from_angle(45))      # Look right 45°
await actions.nod()                               # Nod yes
await actions.gesture(GestureType.THUMBS_UP)      # Thumbs up

# Locomotion (adapts to wheels vs bipedal)
await actions.walk_to(Target.from_position(1.0, 0.0))
await actions.turn(angle=90)

# Compound actions
await actions.greet(target=person)                # Look + wave
```

### Adding a New Robot

1. Create adapter in `adapters/your_robot.py` implementing `RobotAdapter`
2. Add camera support in `cameras/` if needed (or use existing OAK-D/ZED)
3. Add microphone support in `microphones/` if needed
4. Register in `config/robots.py`

```python
# adapters/your_robot.py
class YourRobotAdapter(RobotAdapter):
    async def connect(self) -> bool: ...
    async def execute(self, subsystem, action) -> ActionResult: ...
    def get_capabilities(self) -> Dict[str, Any]: ...
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Identity Layer (Gun.js)                                │
│  Face embeddings, voice embeddings, memories            │
│  Syncs across all bodies                                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Perception Layer                                       │
│  cameras/    → SpatialCamera (OAK-D, ZED, RealSense)   │
│  microphones/→ MicrophoneArray (ReSpeaker, 6-Mic)      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Adapter Layer                                          │
│  adapters/   → RobotAdapter (Johnny5, BoosterK1, ...)  │
│  Translates intents to hardware-specific commands       │
└─────────────────────────────────────────────────────────┘
```

See [docs/AUTONOMIC_ARCHITECTURE.md](docs/AUTONOMIC_ARCHITECTURE.md) for details.

---

## Key Files

| File | Purpose |
|------|---------|
| `robot_factory.py` | Unified entry point - creates robot with adapter, camera, mic |
| `johnny5.py` | Main entry - Hume EVI WebSocket + face recognition triggers |
| `motion_coordinator.py` | The "spine" - coordinates all movement, gestures, safety |
| `adapters/base.py` | Abstract RobotAdapter interface (implement for new robots) |
| `adapters/booster_k1.py` | Booster K1 adapter (ROS2 bipedal) |
| `adapters/unitree_g1.py` | Unitree G1 adapter (unitree_sdk2/ROS2) |
| `actions/unified.py` | Unified action primitives (wave, point, walk, etc.) |
| `cameras/base.py` | Abstract SpatialCamera interface (OAK-D, ZED, etc.) |
| `microphones/base.py` | Abstract MicrophoneArray interface (DOA, VAD) |
| `config/robots.py` | Multi-robot configuration registry |
| `tools/registry.py` | 50+ tools Hume can call (wave, point, move, etc.) |
| `doa_spatial_fusion.py` | DOA + depth fusion for active speaker identification |
| `terrain_navigation.py` | Autonomous obstacle handling (gaps, rails, cords) |
| `visual_safety.py` | Fire/smoke detection with auto-alert |

---

## Documentation

- [AUTONOMIC_ARCHITECTURE.md](docs/AUTONOMIC_ARCHITECTURE.md) — Hume ↔ Spine ↔ Adapter with mermaid diagrams
- [JOHNNY5_HARDWARE_SPEC.md](docs/JOHNNY5_HARDWARE_SPEC.md) — 19-servo layout, calibration, motor IDs
- [HUME_EVI_ROBOT_CONTROL_PLAN.md](docs/HUME_EVI_ROBOT_CONTROL_PLAN.md) — Tool execution architecture
- [SENSOR_UTILIZATION.md](docs/SENSOR_UTILIZATION.md) — ReSpeaker & OAK-D capabilities and usage
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
