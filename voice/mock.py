"""Mock voice backends for testing.

These do nothing but satisfy the interface.
Useful for:
- Running without hardware
- Unit tests
- Last resort in failover chain
"""

import asyncio
from typing import AsyncIterator, Dict, List, Optional

from .base import TTSBackend, STTBackend, LLMBackend, VoiceStack


class MockTTS(TTSBackend):
    """Mock TTS that logs but doesn't speak."""

    def __init__(self):
        self._speaking = False

    @property
    def name(self) -> str:
        return "mock-tts"

    async def speak(self, text: str) -> None:
        """Log the text instead of speaking."""
        print(f"[MockTTS] Would speak: {text}")
        self._speaking = True
        await asyncio.sleep(0.1)  # Simulate speaking time
        self._speaking = False

    async def speak_stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield empty bytes."""
        await self.speak(text)
        yield b""

    def is_speaking(self) -> bool:
        return self._speaking

    async def stop(self) -> None:
        self._speaking = False


class MockSTT(STTBackend):
    """Mock STT that yields predefined transcripts."""

    def __init__(self, transcripts: Optional[List[str]] = None):
        self._transcripts = transcripts or []
        self._listening = False
        self._index = 0

    @property
    def name(self) -> str:
        return "mock-stt"

    async def start(self) -> None:
        self._listening = True

    async def stop(self) -> None:
        self._listening = False

    async def transcripts(self) -> AsyncIterator[str]:
        """Yield predefined transcripts."""
        while self._listening and self._index < len(self._transcripts):
            yield self._transcripts[self._index]
            self._index += 1
            await asyncio.sleep(1)

    def is_listening(self) -> bool:
        return self._listening

    def add_transcript(self, text: str) -> None:
        """Add a transcript to the queue (for testing)."""
        self._transcripts.append(text)


class MockLLM(LLMBackend):
    """Mock LLM that echoes or returns predefined responses."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self._responses = responses or {}
        self._system_prompt = ""
        self._default_response = "I am a mock response."

    @property
    def name(self) -> str:
        return "mock-llm"

    async def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Return predefined response or echo."""
        if message in self._responses:
            return self._responses[message]
        return f"[MockLLM] You said: {message}"

    async def chat_stream(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> AsyncIterator[str]:
        """Stream response word by word."""
        response = await self.chat(message, history)
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.05)

    async def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def set_response(self, trigger: str, response: str) -> None:
        """Set a predefined response for a trigger phrase."""
        self._responses[trigger] = response


def create_mock_stack(
    transcripts: Optional[List[str]] = None,
    responses: Optional[Dict[str, str]] = None,
) -> VoiceStack:
    """Create a mock voice stack for testing.

    Args:
        transcripts: List of transcripts the STT will yield
        responses: Dict mapping user messages to LLM responses

    Returns:
        VoiceStack with mock backends
    """
    return VoiceStack(
        tts=MockTTS(),
        stt=MockSTT(transcripts),
        llm=MockLLM(responses),
    )
