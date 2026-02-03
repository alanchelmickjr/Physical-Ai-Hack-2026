"""Abstract base classes for voice components.

Each component (TTS, STT, LLM) has a clear interface that both
tethered (cloud) and local implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Optional, List, Dict, Any
import asyncio


class VoiceEvent(Enum):
    """Events emitted by voice components."""
    LISTENING_START = "listening_start"
    LISTENING_STOP = "listening_stop"
    SPEAKING_START = "speaking_start"
    SPEAKING_STOP = "speaking_stop"
    THINKING_START = "thinking_start"
    THINKING_STOP = "thinking_stop"
    TRANSCRIPT = "transcript"
    ERROR = "error"


@dataclass
class VoiceMessage:
    """Message passed between voice components."""
    event: VoiceEvent
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class TTSBackend(ABC):
    """Abstract Text-to-Speech backend.

    Implementations:
    - HumeTTS: Cloud-based with emotion
    - KokoroTTS: Local neural TTS
    - PiperTTS: Local lightweight TTS
    - MockTTS: For testing
    """

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Speak text asynchronously. Blocks until complete."""
        pass

    @abstractmethod
    async def speak_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks for text. Yields PCM audio bytes."""
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """Return True if currently speaking."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop current speech immediately."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass

    @property
    def is_local(self) -> bool:
        """Return True if this is a local (non-cloud) backend."""
        return True


class STTBackend(ABC):
    """Abstract Speech-to-Text backend.

    Implementations:
    - HumeSTT: Cloud-based with emotion detection
    - VoskSTT: Local offline recognition
    - WhisperSTT: Local Whisper model
    - MockSTT: For testing
    """

    @abstractmethod
    async def start(self) -> None:
        """Start listening for speech."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening."""
        pass

    @abstractmethod
    async def transcripts(self) -> AsyncIterator[str]:
        """Yield transcribed text as it's recognized."""
        pass

    @abstractmethod
    def is_listening(self) -> bool:
        """Return True if currently listening."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass

    @property
    def is_local(self) -> bool:
        """Return True if this is a local (non-cloud) backend."""
        return True


class LLMBackend(ABC):
    """Abstract LLM backend for conversation.

    Implementations:
    - HumeLLM: Cloud-based empathic AI
    - OllamaLLM: Local Ollama models
    - MockLLM: For testing
    """

    @abstractmethod
    async def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Send message and get response."""
        pass

    @abstractmethod
    async def chat_stream(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> AsyncIterator[str]:
        """Stream response tokens."""
        pass

    @abstractmethod
    async def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for conversation."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass

    @property
    def is_local(self) -> bool:
        """Return True if this is a local (non-cloud) backend."""
        return True


@dataclass
class VoiceStack:
    """Complete voice stack with TTS, STT, and LLM.

    This is what the factory returns. Components can be mixed:
    - Full Hume (tethered)
    - Full local (Kokoro + Vosk + Ollama)
    - Hybrid (e.g., local STT + cloud LLM)
    """
    tts: TTSBackend
    stt: STTBackend
    llm: LLMBackend

    # Event callbacks
    _event_handlers: Dict[VoiceEvent, List[Callable]] = field(default_factory=dict)
    _running: bool = False

    def on(self, event: VoiceEvent, handler: Callable) -> None:
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def emit(self, event: VoiceEvent, text: str = "", **data) -> None:
        """Emit event to handlers."""
        msg = VoiceMessage(event=event, text=text, data=data)
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(msg))
                else:
                    handler(msg)
            except Exception as e:
                print(f"Event handler error: {e}")

    async def conversation_loop(self) -> None:
        """Main conversation loop - listen, think, speak."""
        self._running = True

        while self._running:
            try:
                # Listen
                self.emit(VoiceEvent.LISTENING_START)
                await self.stt.start()

                async for transcript in self.stt.transcripts():
                    if not transcript.strip():
                        continue

                    self.emit(VoiceEvent.TRANSCRIPT, transcript)
                    self.emit(VoiceEvent.LISTENING_STOP)

                    # Think
                    self.emit(VoiceEvent.THINKING_START)
                    response = await self.llm.chat(transcript)
                    self.emit(VoiceEvent.THINKING_STOP)

                    # Speak
                    self.emit(VoiceEvent.SPEAKING_START, response)
                    await self.tts.speak(response)
                    self.emit(VoiceEvent.SPEAKING_STOP)

                    # Resume listening
                    self.emit(VoiceEvent.LISTENING_START)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.emit(VoiceEvent.ERROR, str(e))
                await asyncio.sleep(1)

        await self.stt.stop()

    async def stop(self) -> None:
        """Stop the conversation loop."""
        self._running = False
        await self.stt.stop()
        await self.tts.stop()

    @property
    def is_local(self) -> bool:
        """Return True if all components are local."""
        return self.tts.is_local and self.stt.is_local and self.llm.is_local

    def status(self) -> Dict[str, str]:
        """Return status of all components."""
        return {
            "tts": self.tts.name,
            "stt": self.stt.name,
            "llm": self.llm.name,
            "mode": "local" if self.is_local else "tethered",
        }
