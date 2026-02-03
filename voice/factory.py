"""Voice factory with automatic failover.

Creates voice stacks with failure chain:
1. Try tethered (Hume) first
2. If unavailable/fails → fall back to local (Chloe)
3. If local fails → fall back to mock (silent)

Supports dependency injection for testing.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Type, Dict, Any, Callable

from .base import VoiceStack, TTSBackend, STTBackend, LLMBackend

logger = logging.getLogger(__name__)


class VoiceType(Enum):
    """Voice stack types."""
    HUME = "hume"           # Cloud EVI (tethered)
    LOCAL = "local"         # Kokoro + Vosk + Ollama
    LOCAL_PIPER = "piper"   # Piper + Vosk + Ollama (lighter)
    MOCK = "mock"           # Silent (for testing)
    AUTO = "auto"           # Try Hume, fallback to local


class FailureReason(Enum):
    """Why a backend failed."""
    NOT_INSTALLED = "not_installed"
    CREDITS_EXHAUSTED = "credits_exhausted"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    HARDWARE_MISSING = "hardware_missing"
    UNKNOWN = "unknown"


@dataclass
class FailoverEvent:
    """Record of a failover event."""
    from_backend: str
    to_backend: str
    reason: FailureReason
    error_message: str
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class FactoryConfig:
    """Configuration for voice factory."""
    # Hume settings
    hume_api_key: Optional[str] = None
    hume_config_id: Optional[str] = None

    # Local TTS settings
    tts_backend: str = "kokoro"  # "kokoro" or "piper"
    tts_voice: str = "af_heart"
    piper_dir: Optional[str] = None
    piper_model: Optional[str] = None

    # Local STT settings
    vosk_model_path: Optional[str] = None
    audio_device: Optional[int] = None

    # Local LLM settings
    llm_model: str = "qwen2.5:1.5b"
    ollama_host: Optional[str] = None

    # System prompt (shared across backends)
    system_prompt: str = ""

    # Failover settings
    auto_failover: bool = True
    failover_timeout: float = 10.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "FactoryConfig":
        """Create config from environment variables."""
        return cls(
            hume_api_key=os.getenv("HUME_API_KEY"),
            hume_config_id=os.getenv("HUME_CONFIG_ID"),
            tts_backend=os.getenv("TTS_BACKEND", "kokoro"),
            llm_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            system_prompt=os.getenv("SYSTEM_PROMPT", ""),
            auto_failover=os.getenv("AUTO_FAILOVER", "1").lower() in ("1", "true"),
        )


class VoiceFactory:
    """Factory for creating voice stacks with failover support.

    Usage:
        factory = VoiceFactory(config)
        stack = await factory.create(VoiceType.AUTO)

        # Or with manual failover
        factory = VoiceFactory(config, auto_failover=False)
        stack = await factory.create(VoiceType.HUME)
        if stack is None:
            stack = await factory.create(VoiceType.LOCAL)
    """

    # Failover chain: if one fails, try next
    FAILOVER_CHAIN = [
        VoiceType.HUME,
        VoiceType.LOCAL,
        VoiceType.LOCAL_PIPER,
        VoiceType.MOCK,
    ]

    def __init__(self, config: Optional[FactoryConfig] = None):
        self.config = config or FactoryConfig.from_env()
        self._failover_history: List[FailoverEvent] = []
        self._current_stack: Optional[VoiceStack] = None
        self._current_type: Optional[VoiceType] = None
        self._failover_handlers: List[Callable[[FailoverEvent], None]] = []

    def on_failover(self, handler: Callable[[FailoverEvent], None]) -> None:
        """Register a handler for failover events."""
        self._failover_handlers.append(handler)

    def _emit_failover(self, event: FailoverEvent) -> None:
        """Emit failover event to handlers."""
        self._failover_history.append(event)
        for handler in self._failover_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Failover handler error: {e}")

    async def create(self, voice_type: VoiceType = VoiceType.AUTO) -> VoiceStack:
        """Create a voice stack, with automatic failover if enabled.

        Args:
            voice_type: Type of stack to create (AUTO for automatic selection)

        Returns:
            VoiceStack instance

        Raises:
            RuntimeError: If no backends are available
        """
        if voice_type == VoiceType.AUTO:
            return await self._create_with_failover()

        return await self._create_specific(voice_type)

    async def _create_with_failover(self) -> VoiceStack:
        """Try backends in order until one works."""
        last_error = None

        for voice_type in self.FAILOVER_CHAIN:
            try:
                stack = await self._create_specific(voice_type)
                if stack:
                    logger.info(f"Voice stack created: {voice_type.value}")
                    self._current_stack = stack
                    self._current_type = voice_type
                    return stack
            except Exception as e:
                reason = self._classify_error(e)
                last_error = e
                logger.warning(f"{voice_type.value} failed ({reason.value}): {e}")

                # Record failover
                next_type = self._get_next_in_chain(voice_type)
                if next_type:
                    self._emit_failover(FailoverEvent(
                        from_backend=voice_type.value,
                        to_backend=next_type.value,
                        reason=reason,
                        error_message=str(e),
                    ))

        raise RuntimeError(f"All voice backends failed. Last error: {last_error}")

    async def _create_specific(self, voice_type: VoiceType) -> VoiceStack:
        """Create a specific voice stack type."""
        if voice_type == VoiceType.HUME:
            return await self._create_hume()
        elif voice_type == VoiceType.LOCAL:
            return self._create_local(tts="kokoro")
        elif voice_type == VoiceType.LOCAL_PIPER:
            return self._create_local(tts="piper")
        elif voice_type == VoiceType.MOCK:
            return self._create_mock()
        else:
            raise ValueError(f"Unknown voice type: {voice_type}")

    async def _create_hume(self) -> VoiceStack:
        """Create Hume EVI stack."""
        from .hume import create_hume_stack

        if not self.config.hume_api_key:
            raise ValueError("HUME_API_KEY not configured")

        stack = create_hume_stack(
            api_key=self.config.hume_api_key,
            config_id=self.config.hume_config_id,
            system_prompt=self.config.system_prompt,
        )

        # Test connection
        hume = stack.tts  # TTS/STT/LLM are same instance for Hume
        await hume.connect()

        return stack

    def _create_local(self, tts: str = "kokoro") -> VoiceStack:
        """Create local voice stack."""
        from .local import create_local_stack

        return create_local_stack(
            tts_backend=tts,
            stt_backend="vosk",
            llm_backend="ollama",
            llm_model=self.config.llm_model,
            system_prompt=self.config.system_prompt,
        )

    def _create_mock(self) -> VoiceStack:
        """Create mock voice stack for testing."""
        from .mock import create_mock_stack
        return create_mock_stack()

    def _classify_error(self, error: Exception) -> FailureReason:
        """Classify an error to determine failover reason."""
        error_str = str(error).lower()

        if "not installed" in error_str or "no module" in error_str:
            return FailureReason.NOT_INSTALLED
        elif "credit" in error_str or "balance" in error_str or "billing" in error_str:
            return FailureReason.CREDITS_EXHAUSTED
        elif "connection" in error_str or "connect" in error_str:
            return FailureReason.CONNECTION_ERROR
        elif "timeout" in error_str:
            return FailureReason.TIMEOUT
        elif "device" in error_str or "hardware" in error_str:
            return FailureReason.HARDWARE_MISSING
        else:
            return FailureReason.UNKNOWN

    def _get_next_in_chain(self, current: VoiceType) -> Optional[VoiceType]:
        """Get next backend in failover chain."""
        try:
            idx = self.FAILOVER_CHAIN.index(current)
            if idx + 1 < len(self.FAILOVER_CHAIN):
                return self.FAILOVER_CHAIN[idx + 1]
        except ValueError:
            pass
        return None

    @property
    def current_type(self) -> Optional[VoiceType]:
        """Get the currently active voice type."""
        return self._current_type

    @property
    def failover_history(self) -> List[FailoverEvent]:
        """Get history of failover events."""
        return list(self._failover_history)

    def status(self) -> Dict[str, Any]:
        """Get factory status."""
        return {
            "current_type": self._current_type.value if self._current_type else None,
            "stack_status": self._current_stack.status() if self._current_stack else None,
            "failover_count": len(self._failover_history),
            "config": {
                "auto_failover": self.config.auto_failover,
                "hume_configured": bool(self.config.hume_api_key),
                "llm_model": self.config.llm_model,
            }
        }


# Singleton instance
_factory: Optional[VoiceFactory] = None


def get_factory(config: Optional[FactoryConfig] = None) -> VoiceFactory:
    """Get or create the global voice factory."""
    global _factory
    if _factory is None or config is not None:
        _factory = VoiceFactory(config)
    return _factory


async def create_voice_stack(
    voice_type: VoiceType = VoiceType.AUTO,
    config: Optional[FactoryConfig] = None,
) -> VoiceStack:
    """Convenience function to create a voice stack.

    Args:
        voice_type: Type of stack (AUTO, HUME, LOCAL, MOCK)
        config: Optional configuration override

    Returns:
        VoiceStack ready for use
    """
    factory = get_factory(config)
    return await factory.create(voice_type)


def get_voice_stack() -> Optional[VoiceStack]:
    """Get the current voice stack if one exists."""
    if _factory and _factory._current_stack:
        return _factory._current_stack
    return None
