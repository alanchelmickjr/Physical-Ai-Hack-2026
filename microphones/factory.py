"""Microphone Factory

Creates the appropriate microphone instance based on type.
Register custom microphones here.
"""

from typing import Dict, Type, Optional

from .base import MicrophoneArray, MicrophoneType


# Microphone registry
_MIC_REGISTRY: Dict[MicrophoneType, Type[MicrophoneArray]] = {}


def register_microphone(mic_type: MicrophoneType, mic_class: Type[MicrophoneArray]):
    """Register a microphone implementation.

    Args:
        mic_type: MicrophoneType enum value
        mic_class: MicrophoneArray subclass
    """
    _MIC_REGISTRY[mic_type] = mic_class
    print(f"[MicFactory] Registered {mic_type.value}: {mic_class.__name__}")


def get_microphone(mic_type: MicrophoneType) -> MicrophoneArray:
    """Get a microphone instance for the given type.

    Args:
        mic_type: Type of microphone to create

    Returns:
        MicrophoneArray instance

    Raises:
        ValueError: If microphone type not registered
    """
    _ensure_registered()

    if mic_type not in _MIC_REGISTRY:
        raise ValueError(f"Unknown microphone type: {mic_type}. "
                        f"Available: {list(_MIC_REGISTRY.keys())}")

    mic_class = _MIC_REGISTRY[mic_type]
    return mic_class(mic_type)


def _ensure_registered():
    """Ensure default microphones are registered."""
    if _MIC_REGISTRY:
        return

    try:
        from .respeaker import ReSpeakerMic
        register_microphone(MicrophoneType.RESPEAKER_4MIC, ReSpeakerMic)
        register_microphone(MicrophoneType.RESPEAKER_6MIC, ReSpeakerMic)
    except ImportError as e:
        print(f"[MicFactory] ReSpeaker not available: {e}")

    try:
        from .circular6 import Circular6Mic
        register_microphone(MicrophoneType.CIRCULAR_6MIC, Circular6Mic)
    except ImportError as e:
        print(f"[MicFactory] Circular6Mic not available: {e}")


def list_available_microphones() -> list:
    """List all registered microphone types.

    Returns:
        List of available MicrophoneType values
    """
    _ensure_registered()
    return list(_MIC_REGISTRY.keys())
