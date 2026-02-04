"""Hume EVI backend - Tethered cloud voice with empathic AI.

This is the "Johnny Five" voice - cloud-based, expressive, emotion-aware.
Falls back to local if credits exhausted or connection fails.

Supports:
- Hot-loadable config IDs (personality + tools)
- Tool call handling with local execution engine
- Session settings updates
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import AsyncIterator, Callable, Dict, List, Optional

from .base import TTSBackend, STTBackend, LLMBackend

logger = logging.getLogger(__name__)

# Lazy imports - only load Hume SDK when needed
_hume_available = None


def _check_hume():
    """Check if Hume SDK is available."""
    global _hume_available
    if _hume_available is None:
        try:
            from hume import MicrophoneInterface, Stream
            from hume.client import AsyncHumeClient
            from hume.empathic_voice.types import SubscribeEvent, UserInput, SessionSettings
            _hume_available = True
        except ImportError:
            _hume_available = False
    return _hume_available


class HumeVoice(TTSBackend, STTBackend, LLMBackend):
    """Unified Hume EVI backend - handles TTS, STT, and LLM as one unit.

    Hume EVI is a unified voice system, not separate components.
    This class presents the same interface as separate backends
    but internally uses Hume's websocket connection.

    Features:
    - Hot-loadable config IDs (personality + tools bundled)
    - Tool call handling - receives tool_call, executes locally, sends tool_response
    - Session settings updates for system prompt changes
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tool_handler: Optional[Callable] = None,
    ):
        if not _check_hume():
            raise RuntimeError("Hume SDK not installed. Install with: pip install hume")

        self.api_key = api_key or os.getenv("HUME_API_KEY")
        self.config_id = config_id or os.getenv("HUME_CONFIG_ID")
        self.system_prompt = system_prompt or ""

        # Tool handler callback - called when Hume sends a tool_call
        # Signature: async def handler(name: str, args: dict, tool_call_id: str) -> dict
        self._tool_handler = tool_handler

        if not self.api_key:
            raise ValueError("HUME_API_KEY required")

        self._socket = None
        self._stream = None
        self._client = None
        self._speaking = False
        self._listening = False
        self._connected = False
        self._transcript_queue: asyncio.Queue = asyncio.Queue()
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._tool_call_queue: asyncio.Queue = asyncio.Queue()
        self._message_task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        return "hume-evi"

    @property
    def is_local(self) -> bool:
        return False

    async def connect(self) -> None:
        """Establish Hume websocket connection."""
        if self._connected:
            return

        from hume import Stream
        from hume.client import AsyncHumeClient
        from hume.empathic_voice.types import SessionSettings

        self._client = AsyncHumeClient(api_key=self.api_key)
        self._stream = Stream.new()

        connect_kwargs = {}
        if self.config_id:
            connect_kwargs["config_id"] = self.config_id

        self._socket = await self._client.empathic_voice.chat.connect(**connect_kwargs).__aenter__()
        self._connected = True

        # Set system prompt if provided
        if self.system_prompt:
            await self._socket.send_session_settings(
                SessionSettings(system_prompt=self.system_prompt)
            )

        # Start message handler
        self._message_task = asyncio.create_task(self._handle_messages())

    async def disconnect(self) -> None:
        """Close Hume connection."""
        if self._message_task:
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass

        if self._socket:
            await self._socket.__aexit__(None, None, None)
            self._socket = None

        self._connected = False

    async def _handle_messages(self) -> None:
        """Handle incoming Hume messages including tool calls."""
        try:
            async for message in self._socket:
                msg_type = getattr(message, 'type', None)

                if msg_type == "user_message":
                    # Transcription received
                    if hasattr(message, 'message') and hasattr(message.message, 'content'):
                        await self._transcript_queue.put(message.message.content)

                elif msg_type == "assistant_message":
                    # LLM response
                    if hasattr(message, 'message') and hasattr(message.message, 'content'):
                        await self._response_queue.put(message.message.content)

                elif msg_type == "audio_output":
                    # TTS audio chunk
                    self._speaking = True
                    if hasattr(message, 'data'):
                        audio_bytes = base64.b64decode(message.data)
                        await self._audio_queue.put(audio_bytes)

                elif msg_type == "assistant_end":
                    # Done speaking
                    self._speaking = False
                    await self._audio_queue.put(None)  # Signal end

                elif msg_type == "tool_call":
                    # Hume is requesting tool execution
                    await self._handle_tool_call(message)

                elif msg_type == "error":
                    error_msg = getattr(message, 'message', str(message))
                    logger.error(f"Hume error: {error_msg}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Message handler error: {e}")

    async def _handle_tool_call(self, message) -> None:
        """Handle a tool_call message from Hume."""
        tool_name = getattr(message, 'name', None)
        tool_call_id = getattr(message, 'tool_call_id', None)
        parameters = getattr(message, 'parameters', '{}')

        if not tool_name or not tool_call_id:
            logger.error(f"Invalid tool_call message: {message}")
            return

        # Parse parameters
        try:
            args = json.loads(parameters) if isinstance(parameters, str) else parameters
        except json.JSONDecodeError:
            args = {}

        logger.info(f"Tool call: {tool_name}({args}) id={tool_call_id}")

        # Execute via handler if registered
        if self._tool_handler:
            try:
                result = await self._tool_handler(tool_name, args, tool_call_id)

                # Send tool response back to Hume
                await self._socket.send_tool_response(
                    tool_call_id=tool_call_id,
                    content=json.dumps(result) if isinstance(result, dict) else str(result)
                )
                logger.info(f"Tool result: {result}")

            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                # Send error response
                await self._socket.send_tool_error(
                    tool_call_id=tool_call_id,
                    error=str(e),
                    fallback_content=f"Tool {tool_name} failed: {e}"
                )
        else:
            # No handler registered - send error
            logger.warning(f"No tool handler for: {tool_name}")
            await self._socket.send_tool_error(
                tool_call_id=tool_call_id,
                error="No tool handler registered",
                fallback_content=f"I don't know how to {tool_name}"
            )

    def set_tool_handler(self, handler: Callable) -> None:
        """Set the tool execution handler.

        Handler signature: async def handler(name: str, args: dict, tool_call_id: str) -> dict
        """
        self._tool_handler = handler

    async def switch_config(self, config_id: str) -> None:
        """Hot-switch to a different Hume config ID.

        This allows changing personality + tools at runtime.
        Requires reconnecting to Hume with the new config.
        """
        if config_id == self.config_id:
            return

        logger.info(f"Switching Hume config: {self.config_id} -> {config_id}")

        # Disconnect current session
        await self.disconnect()

        # Update config and reconnect
        self.config_id = config_id
        await self.connect()

    # TTSBackend implementation

    async def speak(self, text: str) -> None:
        """Hume doesn't support direct TTS - it's conversational.
        This sends the text as a user input to trigger a response.
        """
        if not self._connected:
            await self.connect()

        from hume.empathic_voice.types import UserInput
        await self._socket.send_publish(UserInput(text=text))

        # Wait for response to complete
        while self._speaking:
            await asyncio.sleep(0.1)

    async def speak_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio from Hume response."""
        if not self._connected:
            await self.connect()

        from hume.empathic_voice.types import UserInput
        await self._socket.send_publish(UserInput(text=text))

        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def is_speaking(self) -> bool:
        return self._speaking

    async def stop(self) -> None:
        """Stop current speech."""
        # Hume doesn't have a direct stop - would need to disconnect
        pass

    # STTBackend implementation

    async def start(self) -> None:
        """Start Hume microphone interface."""
        if not self._connected:
            await self.connect()
        self._listening = True

    async def stop_listening(self) -> None:
        """Stop listening."""
        self._listening = False

    async def transcripts(self) -> AsyncIterator[str]:
        """Yield transcripts from Hume."""
        while self._listening:
            try:
                transcript = await asyncio.wait_for(
                    self._transcript_queue.get(),
                    timeout=0.5
                )
                yield transcript
            except asyncio.TimeoutError:
                continue

    def is_listening(self) -> bool:
        return self._listening

    # LLMBackend implementation

    async def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Send message and get response via Hume."""
        if not self._connected:
            await self.connect()

        from hume.empathic_voice.types import UserInput
        await self._socket.send_publish(UserInput(text=message))

        # Wait for response
        response = await self._response_queue.get()
        return response

    async def chat_stream(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> AsyncIterator[str]:
        """Hume doesn't stream text tokens - yield full response."""
        response = await self.chat(message, history)
        yield response

    async def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt."""
        self.system_prompt = prompt
        if self._connected:
            from hume.empathic_voice.types import SessionSettings
            await self._socket.send_session_settings(
                SessionSettings(system_prompt=prompt)
            )


def create_hume_stack(
    api_key: Optional[str] = None,
    config_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
):
    """Create a Hume voice stack.

    Returns a VoiceStack where TTS, STT, and LLM all point to
    the same HumeVoice instance (Hume is unified).
    """
    from .base import VoiceStack

    hume = HumeVoice(
        api_key=api_key,
        config_id=config_id,
        system_prompt=system_prompt,
    )

    # Same instance for all - Hume is a unified system
    return VoiceStack(tts=hume, stt=hume, llm=hume)
