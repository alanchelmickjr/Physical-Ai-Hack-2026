"""Local voice backends - No cloud required.

This is "Chloe" - fully local voice stack:
- KokoroTTS or PiperTTS for speech synthesis
- VoskSTT for speech recognition
- OllamaLLM for conversation

All runs on-device, works offline.
"""

import asyncio
import json
import os
import queue
import subprocess
import threading
from typing import AsyncIterator, Dict, List, Optional

from .base import TTSBackend, STTBackend, LLMBackend


# ============================================================
# TTS Backends
# ============================================================

class KokoroTTS(TTSBackend):
    """Kokoro neural TTS - high quality local voice."""

    def __init__(self, voice: str = "af_heart", device: str = "cpu"):
        self._voice = voice
        self._device = device
        self._speaking = False
        self._tts = None
        self._stop_event = asyncio.Event()

        # Force CPU for TTS (save GPU for LLM)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    def _load_tts(self):
        """Lazy load Kokoro TTS."""
        if self._tts is None:
            try:
                from tts_kokoro import KokoroTTS as _Kokoro
                self._tts = _Kokoro(voice=self._voice)
            except ImportError:
                raise RuntimeError("Kokoro TTS not installed. Install with: pip install tts-kokoro")
        return self._tts

    @property
    def name(self) -> str:
        return "kokoro-tts"

    async def speak(self, text: str) -> None:
        """Speak text using Kokoro."""
        if not text.strip():
            return

        self._speaking = True
        self._stop_event.clear()

        try:
            tts = self._load_tts()
            # Run in thread to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, tts.say, text)
        finally:
            self._speaking = False

    async def speak_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream not directly supported - yield after generation."""
        # Kokoro generates then plays - would need modification for true streaming
        await self.speak(text)
        yield b""

    def is_speaking(self) -> bool:
        return self._speaking

    async def stop(self) -> None:
        """Stop speaking."""
        self._stop_event.set()
        self._speaking = False


class PiperTTS(TTSBackend):
    """Piper TTS - lightweight local voice."""

    def __init__(
        self,
        piper_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        length_scale: float = 1.1,
    ):
        self._piper_dir = piper_dir or os.path.expanduser("~/piper")
        self._model_path = model_path or os.path.expanduser("~/piper_voices/en_US-ryan-medium.onnx")
        self._length_scale = length_scale
        self._speaking = False
        self._process: Optional[subprocess.Popen] = None

    @property
    def name(self) -> str:
        return "piper-tts"

    async def speak(self, text: str) -> None:
        """Speak using Piper binary."""
        if not text.strip():
            return

        self._speaking = True

        try:
            # Generate speech
            tmp_speech = "/tmp/chloe_speech.wav"
            tmp_final = "/tmp/chloe_final.wav"

            gen_cmd = (
                f'cd {self._piper_dir} && echo "{text}" | '
                f'LD_LIBRARY_PATH=. ./piper --model {self._model_path} '
                f'--length-scale {self._length_scale} --output_file {tmp_speech}'
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(gen_cmd, shell=True, timeout=15, stderr=subprocess.DEVNULL)
            )

            # Add silence prefix for HDMI warmup
            pad_cmd = (
                f'sox -n -r 22050 -c 1 /tmp/silence.wav trim 0.0 0.5 2>/dev/null; '
                f'sox /tmp/silence.wav {tmp_speech} {tmp_final} 2>/dev/null'
            )
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(pad_cmd, shell=True, timeout=5, stderr=subprocess.DEVNULL)
            )

            # Play with PulseAudio
            self._process = subprocess.Popen(
                ["paplay", tmp_final],
                stderr=subprocess.DEVNULL
            )
            await loop.run_in_executor(None, self._process.wait)

            # Cleanup
            for f in [tmp_speech, tmp_final, "/tmp/silence.wav"]:
                try:
                    os.unlink(f)
                except:
                    pass

        finally:
            self._speaking = False
            self._process = None

    async def speak_stream(self, text: str) -> AsyncIterator[bytes]:
        """Piper doesn't support streaming - speak then yield."""
        await self.speak(text)
        yield b""

    def is_speaking(self) -> bool:
        return self._speaking

    async def stop(self) -> None:
        """Stop current playback."""
        if self._process:
            self._process.terminate()
        self._speaking = False


# ============================================================
# STT Backend
# ============================================================

class VoskSTT(STTBackend):
    """Vosk offline speech recognition."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        device: Optional[int] = None,
    ):
        self._model_path = model_path or self._find_model()
        self._sample_rate = sample_rate
        self._device = device
        self._listening = False
        self._recognizer = None
        self._model = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._transcript_queue: asyncio.Queue = asyncio.Queue()
        self._stream = None
        self._listen_task: Optional[asyncio.Task] = None

    def _find_model(self) -> str:
        """Find Vosk model."""
        paths = [
            os.path.expanduser("~/vosk-model-small-en-us-0.15"),
            "vosk-model-small-en-us-0.15",
            "/usr/share/vosk/model",
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        raise RuntimeError(f"Vosk model not found. Tried: {paths}")

    def _load_model(self):
        """Lazy load Vosk model."""
        if self._model is None:
            try:
                from vosk import Model, KaldiRecognizer
                self._model = Model(self._model_path)
                self._recognizer = KaldiRecognizer(self._model, self._sample_rate)
            except ImportError:
                raise RuntimeError("Vosk not installed. Install with: pip install vosk")
        return self._recognizer

    @property
    def name(self) -> str:
        return "vosk-stt"

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice."""
        self._audio_queue.put(bytes(indata))

    async def start(self) -> None:
        """Start listening."""
        if self._listening:
            return

        import sounddevice as sd

        self._load_model()
        self._listening = True

        # Start audio stream
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=8000,
            device=self._device,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()

        # Start recognition task
        self._listen_task = asyncio.create_task(self._recognize_loop())

    async def _recognize_loop(self) -> None:
        """Process audio and yield transcripts."""
        while self._listening:
            try:
                # Get audio from queue
                data = self._audio_queue.get(timeout=0.1)
                if self._recognizer.AcceptWaveform(data):
                    result = json.loads(self._recognizer.Result())
                    text = result.get("text", "").strip()
                    if text and len(text) > 2:
                        await self._transcript_queue.put(text)
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"VoskSTT error: {e}")
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop listening."""
        self._listening = False

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def transcripts(self) -> AsyncIterator[str]:
        """Yield transcripts."""
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


# ============================================================
# LLM Backend
# ============================================================

class OllamaLLM(LLMBackend):
    """Ollama local LLM backend."""

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        system_prompt: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self._model = model
        self._system_prompt = system_prompt or ""
        self._host = host
        self._client = None

    def _get_client(self):
        """Lazy load Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama
            except ImportError:
                raise RuntimeError("Ollama not installed. Install with: pip install ollama")
        return self._client

    @property
    def name(self) -> str:
        return f"ollama-{self._model}"

    async def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Send message and get response."""
        client = self._get_client()

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        if history:
            messages.extend(history[-10:])  # Keep last 10 turns

        messages.append({"role": "user", "content": message})

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat(model=self._model, messages=messages, stream=False)
        )

        return response["message"]["content"]

    async def chat_stream(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> AsyncIterator[str]:
        """Stream response tokens."""
        client = self._get_client()

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        if history:
            messages.extend(history[-10:])

        messages.append({"role": "user", "content": message})

        loop = asyncio.get_event_loop()

        # Ollama streaming
        def stream_chat():
            for chunk in client.chat(model=self._model, messages=messages, stream=True):
                yield chunk["message"]["content"]

        for token in stream_chat():
            yield token

    async def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt."""
        self._system_prompt = prompt


# ============================================================
# Factory function
# ============================================================

def create_local_stack(
    tts_backend: str = "kokoro",
    stt_backend: str = "vosk",
    llm_backend: str = "ollama",
    llm_model: str = "qwen2.5:1.5b",
    system_prompt: Optional[str] = None,
):
    """Create a fully local voice stack.

    Args:
        tts_backend: "kokoro" or "piper"
        stt_backend: "vosk" (only option currently)
        llm_backend: "ollama" (only option currently)
        llm_model: Ollama model name
        system_prompt: System prompt for LLM

    Returns:
        VoiceStack with local backends
    """
    from .base import VoiceStack

    # TTS
    if tts_backend == "kokoro":
        tts = KokoroTTS()
    elif tts_backend == "piper":
        tts = PiperTTS()
    else:
        raise ValueError(f"Unknown TTS backend: {tts_backend}")

    # STT
    if stt_backend == "vosk":
        stt = VoskSTT()
    else:
        raise ValueError(f"Unknown STT backend: {stt_backend}")

    # LLM
    if llm_backend == "ollama":
        llm = OllamaLLM(model=llm_model, system_prompt=system_prompt)
    else:
        raise ValueError(f"Unknown LLM backend: {llm_backend}")

    return VoiceStack(tts=tts, stt=stt, llm=llm)
