# Modular Component Architecture

This document describes the modular factory-based architecture for Johnny Five's subsystems.

## Design Philosophy

Every major subsystem follows the same pattern:
1. **Abstract Interface** - Defines the contract (`base.py`)
2. **Implementations** - Concrete backends (cloud, local, mock)
3. **Factory** - Creates instances with automatic failover
4. **IPC Channel** - Pub/sub communication with other subsystems

This allows:
- **Hot-swapping** backends without code changes
- **Automatic failover** when tethered services fail
- **Testing** with mock implementations
- **Multi-robot portability** across different hardware

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           IPC MESSAGE BUS                                │
│  Topics: voice.*, vision.*, audio.*, sensor.*, actuator.*, system.*     │
└────────┬──────────────┬──────────────┬──────────────┬──────────────┬────┘
         │              │              │              │              │
    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐    ┌────▼────┐
    │  VOICE  │    │ VISION  │   │  AUDIO  │   │ SENSORS │    │ACTUATORS│
    │ Factory │    │ Channel │   │ Channel │   │ Channel │    │ Channel │
    └────┬────┘    └────┬────┘   └────┬────┘   └────┬────┘    └────┬────┘
         │              │              │              │              │
    ┌────▼────┐    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐    ┌────▼────┐
    │ Hume    │    │ WhoAmI  │   │ReSpeaker│   │  LIDAR  │    │  LED    │
    │   or    │    │  Face   │   │   DOA   │   │  Smell  │    │ Motors  │
    │ Local   │    │  Recog  │   │  Array  │   │  Touch  │    │ Gantry  │
    └─────────┘    └─────────┘   └─────────┘   └─────────┘    └─────────┘
```

---

## Voice Module (`voice/`)

The voice module handles all speech interaction with automatic failover.

### Components

| File | Purpose |
|------|---------|
| `base.py` | Abstract interfaces: `TTSBackend`, `STTBackend`, `LLMBackend`, `VoiceStack` |
| `hume.py` | Hume EVI unified backend (tethered/cloud) |
| `local.py` | Local backends: `KokoroTTS`, `PiperTTS`, `VoskSTT`, `OllamaLLM` |
| `mock.py` | Mock backends for testing |
| `factory.py` | `VoiceFactory` with failover chain |

### Failover Chain

```
HUME (cloud) ─┐
              │ credits exhausted
              ▼
LOCAL/KOKORO ─┐
              │ not installed
              ▼
LOCAL/PIPER ──┐
              │ not installed
              ▼
MOCK (silent)
```

### Usage

```python
from voice import VoiceType, create_voice_stack

# Automatic failover
stack = await create_voice_stack(VoiceType.AUTO)

# Force specific backend
stack = await create_voice_stack(VoiceType.HUME)
stack = await create_voice_stack(VoiceType.LOCAL)

# Check what's running
print(stack.status())
# {'tts': 'hume-evi', 'stt': 'hume-evi', 'llm': 'hume-evi', 'mode': 'tethered'}

# Run conversation loop
await stack.conversation_loop()
```

### Configuration

| Env Variable | Default | Description |
|--------------|---------|-------------|
| `USE_LOCAL` | `0` | Force local mode (`1` = local, `0` = try Hume first) |
| `HUME_API_KEY` | - | Hume API key |
| `HUME_CONFIG_ID` | - | Hume config ID (optional) |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Local LLM model |

### Interfaces

```python
class TTSBackend(ABC):
    async def speak(self, text: str) -> None
    async def speak_stream(self, text: str) -> AsyncIterator[bytes]
    def is_speaking(self) -> bool
    async def stop(self) -> None
    @property
    def name(self) -> str

class STTBackend(ABC):
    async def start(self) -> None
    async def stop(self) -> None
    async def transcripts(self) -> AsyncIterator[str]
    def is_listening(self) -> bool
    @property
    def name(self) -> str

class LLMBackend(ABC):
    async def chat(self, message: str, history: List[Dict]) -> str
    async def chat_stream(self, message: str, history: List[Dict]) -> AsyncIterator[str]
    async def set_system_prompt(self, prompt: str) -> None
    @property
    def name(self) -> str
```

---

## IPC Module (`ipc/`)

The IPC module provides pub/sub messaging between all subsystems.

### Components

| File | Purpose |
|------|---------|
| `bus.py` | `MessageBus` singleton with topic routing |
| `channels.py` | Typed channels: `VoiceChannel`, `VisionChannel`, `AudioChannel`, etc. |

### Topics

| Topic | Publisher | Subscribers |
|-------|-----------|-------------|
| `voice.transcript` | STT | LLM, logging |
| `voice.response` | LLM | TTS, logging |
| `voice.speaking.start` | TTS | LED, state |
| `voice.speaking.stop` | TTS | LED, state |
| `vision.face.recognized` | WhoAmI | Voice, logging |
| `vision.greeting` | WhoAmI | Voice |
| `vision.enroll.request` | Voice | WhoAmI |
| `vision.enroll.complete` | WhoAmI | Voice |
| `audio.doa` | ReSpeaker | Head tracker |
| `audio.vad` | ReSpeaker | Voice |
| `actuator.led` | Any | LED controller |
| `actuator.gantry` | Head tracker | Motors |

### Usage

```python
from ipc import get_bus, Topic, VisionChannel, VoiceChannel

bus = get_bus()

# Typed channel
vision = VisionChannel(bus, source="whoami")
vision.publish_greeting("Hello Alan!")
vision.publish_face_recognized("Alan", confidence=0.95)

# Subscribe to events
def on_greeting(msg):
    print(f"Greeting: {msg.data['text']}")

bus.subscribe(Topic.VISION_GREETING, on_greeting)

# Wildcard subscription
bus.subscribe_pattern("voice.*", lambda msg: print(f"Voice event: {msg.topic}"))
```

### Message Format

```python
@dataclass
class Message:
    topic: Topic
    data: Dict[str, Any]
    timestamp: float
    source: str
    correlation_id: Optional[str]
```

---

## Data Flow

### Face Recognition → Voice Greeting

```
┌──────────────┐     IPC: vision.greeting      ┌──────────────┐
│  whoami_full │ ────────────────────────────► │   johnny5    │
│    .py       │                               │     .py      │
└──────────────┘                               └──────┬───────┘
       │                                              │
       │ VisionChannel                                │ VoiceChannel
       │ .publish_greeting()                          │
       ▼                                              ▼
┌──────────────┐                               ┌──────────────┐
│  IPC Bus     │◄──────────────────────────────│  IPC Bus     │
│  (singleton) │                               │  (singleton) │
└──────────────┘                               └──────────────┘
```

### Voice Enrollment Flow

```
User says "My name is Alan"
         │
         ▼
┌─────────────────┐
│  Voice Stack    │
│  (STT → LLM)    │
└────────┬────────┘
         │ IPC: vision.enroll.request
         ▼
┌─────────────────┐
│   whoami_full   │
│  (face recog)   │
└────────┬────────┘
         │ captures face encoding
         │ IPC: vision.enroll.complete
         ▼
┌─────────────────┐
│  Voice Stack    │
│  (TTS)          │
│  "Nice to meet  │
│   you Alan!"    │
└─────────────────┘
```

### Failover Flow

```
johnny5.py starts
         │
         ▼
┌─────────────────┐
│  VoiceFactory   │
│  .create(AUTO)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    success    ┌─────────────────┐
│  Try HUME       │──────────────►│  Return Hume    │
│  backend        │               │  VoiceStack     │
└────────┬────────┘               └─────────────────┘
         │ failure (credits, network)
         ▼
┌─────────────────┐    success    ┌─────────────────┐
│  Try LOCAL      │──────────────►│  Return Local   │
│  (Kokoro)       │               │  VoiceStack     │
└────────┬────────┘               └─────────────────┘
         │ failure (not installed)
         ▼
┌─────────────────┐    success    ┌─────────────────┐
│  Try LOCAL      │──────────────►│  Return Piper   │
│  (Piper)        │               │  VoiceStack     │
└────────┬────────┘               └─────────────────┘
         │ failure
         ▼
┌─────────────────┐
│  Return MOCK    │
│  (silent)       │
└─────────────────┘
```

---

## Existing Factory Modules

The voice/IPC pattern follows existing factory modules:

### cameras/
| File | Purpose |
|------|---------|
| `base.py` | `SpatialCamera` interface |
| `oakd.py` | OAK-D implementation |
| `zed.py` | ZED camera implementation |
| `realsense.py` | Intel RealSense implementation |
| `factory.py` | Camera factory |

### microphones/
| File | Purpose |
|------|---------|
| `base.py` | `MicrophoneArray` interface |
| `respeaker.py` | ReSpeaker 4-mic implementation |
| `circular6.py` | Generic 6-mic implementation |
| `unitree.py` | Unitree built-in mic implementation |
| `factory.py` | Microphone factory |

### adapters/
| File | Purpose |
|------|---------|
| `base.py` | `RobotAdapter` interface |
| `johnny5.py` | Johnny Five (Solo-CLI) implementation |
| `booster_k1.py` | Booster K1 (ROS2) implementation |
| `unitree_g1.py` | Unitree G1 implementation |

### robot_factory.py
Creates complete robot with all subsystems:
```python
robot = create_robot(RobotType.JOHNNY_FIVE)
# Returns Robot(config, adapter, camera, microphone)
```

---

## Adding a New Voice Backend

1. Create implementation in `voice/`:

```python
# voice/whisper.py
from .base import STTBackend

class WhisperSTT(STTBackend):
    @property
    def name(self) -> str:
        return "whisper-stt"

    async def start(self) -> None:
        # Load Whisper model
        pass

    async def transcripts(self) -> AsyncIterator[str]:
        # Yield transcriptions
        pass
```

2. Add to factory failover chain in `voice/factory.py`:

```python
FAILOVER_CHAIN = [
    VoiceType.HUME,
    VoiceType.LOCAL,       # Kokoro + Vosk
    VoiceType.LOCAL_PIPER, # Piper + Vosk
    VoiceType.WHISPER,     # New: Piper + Whisper
    VoiceType.MOCK,
]
```

---

## File Structure

```
Physical-Ai-Hack-2026/
├── voice/                      # Voice subsystem (NEW)
│   ├── __init__.py
│   ├── base.py                 # TTSBackend, STTBackend, LLMBackend interfaces
│   ├── hume.py                 # Hume EVI (tethered)
│   ├── local.py                # Kokoro, Piper, Vosk, Ollama (local)
│   ├── mock.py                 # Mock backends for testing
│   └── factory.py              # VoiceFactory with failover
│
├── ipc/                        # Inter-process communication (NEW)
│   ├── __init__.py
│   ├── bus.py                  # MessageBus pub/sub
│   └── channels.py             # Typed channels (Voice, Vision, Audio, etc.)
│
├── cameras/                    # Camera subsystem (existing)
│   ├── base.py                 # SpatialCamera interface
│   ├── oakd.py, zed.py, realsense.py
│   └── factory.py
│
├── microphones/                # Microphone subsystem (existing)
│   ├── base.py                 # MicrophoneArray interface
│   ├── respeaker.py, circular6.py, unitree.py
│   └── factory.py
│
├── adapters/                   # Robot adapters (existing)
│   ├── base.py                 # RobotAdapter interface
│   ├── johnny5.py, booster_k1.py, unitree_g1.py
│
├── johnny5.py                  # Main entry point (uses voice factory + IPC)
├── whoami_full.py              # Face recognition (uses IPC)
└── robot_factory.py            # Creates complete robot
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LOCAL` | `0` | `1` = force local voice, `0` = try Hume first |
| `HUME_API_KEY` | - | Hume EVI API key |
| `HUME_CONFIG_ID` | - | Hume config ID |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama model for local LLM |
| `TTS_BACKEND` | `kokoro` | Local TTS: `kokoro` or `piper` |
| `AUTO_FAILOVER` | `1` | Enable automatic failover |

---

## Testing

```bash
# Test with mock backends (no hardware needed)
USE_LOCAL=1 python -c "
from voice import create_voice_stack, VoiceType
import asyncio

async def test():
    stack = await create_voice_stack(VoiceType.MOCK)
    print(stack.status())

asyncio.run(test())
"

# Test IPC bus
python -c "
from ipc import get_bus, Topic, VisionChannel

bus = get_bus()
vision = VisionChannel(bus)

bus.subscribe(Topic.VISION_GREETING, lambda m: print(f'Got: {m.data}'))
vision.publish_greeting('Hello test!')
"
```
