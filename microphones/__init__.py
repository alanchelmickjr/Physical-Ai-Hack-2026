"""Microphone Array Abstraction Layer

Provides a unified interface for different microphone arrays.
Supports DOA (Direction of Arrival) and VAD (Voice Activity Detection).

Usage:
    from microphones import get_microphone, MicrophoneType

    mic = get_microphone(MicrophoneType.CIRCULAR_6MIC)
    mic.start()

    doa, is_speaking, confidence = mic.get_doa()
"""

from .base import MicrophoneArray, MicrophoneType
from .factory import get_microphone, register_microphone

__all__ = [
    'MicrophoneArray',
    'MicrophoneType',
    'get_microphone',
    'register_microphone',
]
