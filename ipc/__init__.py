"""IPC module - Inter-process communication for robot components.

Message bus connecting:
- Voice (TTS/STT/LLM)
- Vision (face recognition, tracking)
- Audio (DOA, speaker ID)
- Sensors (smell, touch, etc.)
- Actuators (LED, motors)

Replaces file-based IPC (/tmp/*.txt) with proper pub/sub.
"""

from .bus import MessageBus, Message, Topic, get_bus
from .channels import (
    VoiceChannel,
    VisionChannel,
    AudioChannel,
    SensorChannel,
    ActuatorChannel,
)

__all__ = [
    "MessageBus",
    "Message",
    "Topic",
    "get_bus",
    "VoiceChannel",
    "VisionChannel",
    "AudioChannel",
    "SensorChannel",
    "ActuatorChannel",
]
