# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## DIRECTIVE: Do Exactly What You Are Told

- Only do what is explicitly requested. Nothing more.
- No presumptions. No assumptions. No additional work.
- If unsure which file to edit, ASK. Do not guess.
- Complete the task directly. Do not research unless asked.
- "readme" means README.md. "claude md" means CLAUDE.md. Listen.

## Project Overview

**Johnny Five** - A **social robot** model whose primary purpose is to **connect with and remember people** for social interaction.

> **Naming Convention**: "Johnny Five" is the robot MODEL (like "iPhone").
> Each unit chooses its own name - this one chose "Chloe".

This is a physical AI robot for the 2026 hackathon with multimodal identity recognition. The robot identifies people through three modalities: face (WhoAmI/DeepFace), voice (WeSpeaker ECAPA-TDNN), and direction of arrival (4-mic DOA fused with YOLO).

### Core Mission
Chloe is designed for **human connection**, not task automation. She:
- Remembers everyone she meets (face + voice embeddings)
- Tracks who is speaking and looks at them
- Engages in natural conversation
- Uses gestures and body language to express herself

Key demo: Put a cap over OAK-D camera - Johnny still knows who's talking by voice alone.

## Hardware Configuration

| Component | Spec |
|-----------|------|
| Brain | Jetson Orin 8GB |
| Camera | OAK-D on 2-DOF gantry |
| Microphone | 4-way directional USB (ReSpeaker v2.0) |
| Arms | Dual SO101 via Solo-CLI |
| Power | Anker Solix 12V direct |

**Bus Layout:**
- ACM0: Left arm (1-6), lift (10), wheels (7-9)
- ACM1: Right arm (1-6), gantry (7-8)

## Tech Stack

- **Python** - Core robot logic
- **Node.js** - Gun.js identity storage
- **ROS2** - Robot middleware (optional)
- **TensorRT** - Model optimization for Jetson

## Development Setup

```bash
# Core Python dependencies
pip install wespeaker silero-vad pyroomacoustics --break-system-packages
pip install depthai opencv-python ultralytics --break-system-packages
pip install solo-cli --break-system-packages

# Gun.js (Node side)
npm install gun

# Download WeSpeaker ECAPA-TDNN model
wget https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn-LM/resolve/main/model.onnx \
    -O models/wespeaker_ecapa.onnx

# Silero VAD auto-downloads on first use
python -c "from silero_vad import load_silero_vad; load_silero_vad(onnx=True)"
```

## Build & Run Commands

```bash
# TensorRT optimization (run on Jetson Orin)
/usr/src/tensorrt/bin/trtexec \
    --onnx=wespeaker_ecapa.onnx \
    --saveEngine=speaker.trt \
    --fp16

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
solo robo --calibrate all
solo robo --teleop
```

## Architecture

### Multimodal Identity Pipeline

```
Audio Input → Silero VAD → WeSpeaker ECAPA-TDNN → 192-d embedding
                              ↓
Video Input → YOLO Detection → DeepFace (WhoAmI) → 512-d embedding
                              ↓
4-mic Array → SRP-PHAT DOA → Pixel mapping → Fusion scoring
                              ↓
                        Gun.js Storage (face + voice embeddings)
```

### Fusion Logic
1. If face + voice + DOA agree → high confidence
2. If face + DOA agree → medium confidence
3. If voice only (blindfolded) → voice confidence
4. Conflicts → trust face, log for review

### Key Modules

| File | Purpose |
|------|---------|
| `src/speaker_id.py` | Voice identification (WeSpeaker ECAPA-TDNN 192-d embeddings) |
| `src/doa_tracker.py` | Direction of arrival from 4-mic array |
| `src/fusion.py` | Audio-visual fusion (DOA + YOLO bounding boxes) |
| `src/identity_store.js` | Gun.js storage for face + voice embeddings |
| `src/main.py` | Integration loop combining all modalities |

### Performance Targets

| Stage | Latency |
|-------|---------|
| Audio capture | 16ms |
| DOA estimation (SRP-PHAT) | 20-40ms |
| Video capture | 33ms (30fps) |
| YOLO inference (OAK-D VPU) | 30-50ms |
| Fusion computation | 2-5ms |
| **Total (parallelized)** | **~70-100ms** |

### Memory Budget (8GB Orin)
- Speaker model: ~200 MB
- ArcFace face model: ~300 MB
- Preprocessing buffers: ~200 MB
- **Total: ~700 MB**, leaving ~5GB for other processes

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Cosine similarity threshold | 0.55 | Balanced FAR/FRR for verification |
| DOA uncertainty (sigma) | ±10° | Typical for 4-mic array |
| Min speech for enrollment | 10 seconds | Accumulated during conversation |
| Speaker separation requirement | >20° | ~1m apart at 3m distance |
| Audio sample rate | 16kHz | Required for all audio processing |

## External Dependencies

- **WeSpeaker**: `github.com/wenet-e2e/wespeaker` - Speaker verification toolkit
- **Silero VAD**: `github.com/snakers4/silero-vad` - Voice activity detection
- **Pyroomacoustics**: DOA algorithms (SRP-PHAT, MUSIC)
- **depthai-ros**: OAK-D ROS2 driver with spatial detection
- **ODAS**: `github.com/introlab/odas` - Professional-grade audio localization (optional)
