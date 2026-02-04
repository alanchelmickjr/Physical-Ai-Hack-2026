"""Voice module - Modular TTS, STT, and LLM components with factory pattern.

Supports tethered (cloud) and local backends with automatic failover.
"""

from .base import TTSBackend, STTBackend, LLMBackend, VoiceStack
from .factory import VoiceType, create_voice_stack, get_voice_stack

__all__ = [
    "TTSBackend",
    "STTBackend",
    "LLMBackend",
    "VoiceStack",
    "VoiceType",
    "create_voice_stack",
    "get_voice_stack",
]
