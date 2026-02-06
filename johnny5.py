#!/usr/bin/env python3
"""Johnny 5 - Hume EVI voice conversation.

Uses parec for audio capture. interrupt=false until AEC is fixed.
"""
import asyncio
import base64
import datetime
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from hume import Stream
from hume.client import AsyncHumeClient
from hume.empathic_voice.types import UserInput, SessionSettings, AudioConfiguration

# LED feedback
try:
    from led_controller import get_led
    LED_AVAILABLE = True
except ImportError:
    LED_AVAILABLE = False
    print("WARNING: LED controller not available")

# IPC bus
try:
    from ipc import get_bus, Topic
    IPC_AVAILABLE = True
except ImportError:
    IPC_AVAILABLE = False
    print("WARNING: IPC bus not available")

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

HUME_API_KEY = os.getenv("HUME_API_KEY")
HUME_CONFIG_ID = os.getenv("HUME_CONFIG_ID")

GREETING_FILE = "/tmp/johnny5_greeting.txt"
GREETING_COOLDOWN = 120.0
CONVERSATION_TIMEOUT = 30.0

# State
evi_last_activity = 0.0
in_conversation = False
audio_playing = False
request_start_time = 0.0
first_audio_time = 0.0
audio_chunk_count = 0
total_audio_bytes = 0

print("Audio: parec (ec_source) → Hume [WebRTC AEC]")


def log(text: str, t0: float = None) -> None:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    ts = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
    if t0:
        delta_ms = (time.time() - t0) * 1000
        print(f"[{ts}] (+{delta_ms:.0f}ms) {text}")
    else:
        print(f"[{ts}] {text}")


class PulseAudioMic:
    """Capture from PulseAudio using parec."""

    def __init__(self, source="ec_source", rate=16000, channels=1):
        self.source = source
        self.rate = rate
        self.channels = channels
        self.process = None
        self.chunk_size = 640  # 20ms at 16kHz mono 16-bit

    async def start_capture(self, socket):
        cmd = [
            "parec",
            f"--device={self.source}",
            "--format=s16le",
            f"--rate={self.rate}",
            f"--channels={self.channels}",
            "--raw",
        ]

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        log(f"Mic started: {self.source} @ {self.rate}Hz")

        try:
            while True:
                chunk = await self.process.stdout.read(self.chunk_size)
                if not chunk:
                    break
                await socket._send(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log(f"Mic error: {e}")
        finally:
            if self.process:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    self.process.kill()


async def play_audio(stream):
    """Play audio using Hume SDK."""
    try:
        from hume.empathic_voice.chat.audio.audio_utilities import play_audio_streaming
        await play_audio_streaming(stream)
    except Exception as e:
        log(f"Playback error: {e}")



async def on_message(message, stream) -> None:
    global evi_last_activity, in_conversation, audio_playing
    global first_audio_time, audio_chunk_count, total_audio_bytes, request_start_time

    if message.type == "chat_metadata":
        log(f"Chat: {message.chat_id}")

    elif message.type == "user_message":
        content = message.message.content if hasattr(message.message, 'content') else str(message)
        log(f"user: {content}")
        evi_last_activity = time.time()
        in_conversation = True
        if LED_AVAILABLE:
            get_led().thinking()

    elif message.type == "assistant_message":
        content = message.message.content if hasattr(message.message, 'content') else str(message)
        log(f"assistant: {content}")
        evi_last_activity = time.time()

    elif message.type == "audio_output":
        chunk_bytes = base64.b64decode(message.data.encode("utf-8"))

        if audio_chunk_count == 0:
            if LED_AVAILABLE:
                get_led().speaking()
            audio_playing = True
            first_audio_time = time.time()
            if request_start_time:
                latency_ms = (first_audio_time - request_start_time) * 1000
                log(f"First audio: {latency_ms:.0f}ms latency")

        audio_chunk_count += 1
        total_audio_bytes += len(chunk_bytes)
        await stream.put(chunk_bytes)
        evi_last_activity = time.time()

    elif message.type == "assistant_end":
        if audio_chunk_count > 0:
            duration_ms = (time.time() - first_audio_time) * 1000 if first_audio_time else 0
            log(f"Audio done: {audio_chunk_count} chunks, {total_audio_bytes} bytes, {duration_ms:.0f}ms")

        audio_chunk_count = 0
        total_audio_bytes = 0
        first_audio_time = 0.0
        audio_playing = False

        if LED_AVAILABLE:
            get_led().listening()

    elif message.type == "error":
        log(f"ERROR: {message.code} - {message.message}")
        if LED_AVAILABLE:
            get_led().error()


# IPC greeting queue
_greeting_queue = asyncio.Queue()


def setup_ipc():
    if not IPC_AVAILABLE:
        return
    bus = get_bus()

    def on_greeting(msg):
        text = msg.data.get("text", "")
        if text:
            try:
                _greeting_queue.put_nowait(text)
            except asyncio.QueueFull:
                pass

    bus.subscribe(Topic.VISION_GREETING, on_greeting)
    log("IPC ready")


_last_greeted = {}


async def wait_for_greeting():
    """Poll IPC queue and greeting file. Returns greeting text. No Hume connection."""
    while True:
        # Check IPC queue
        try:
            greeting = _greeting_queue.get_nowait()
        except asyncio.QueueEmpty:
            greeting = None

        # Check file fallback
        if not greeting and os.path.exists(GREETING_FILE):
            with open(GREETING_FILE, 'r') as f:
                greeting = f.read().strip()
            if greeting:
                os.remove(GREETING_FILE)

        if greeting:
            now = time.time()
            words = greeting.split()
            name = words[1].rstrip('!') if len(words) > 1 else "someone"

            if name in _last_greeted and (now - _last_greeted[name]) < GREETING_COOLDOWN:
                log(f"(skip {name} - recent)")
            else:
                _last_greeted[name] = now
                return greeting

        await asyncio.sleep(0.5)


async def check_greetings_during_conversation(socket):
    """Handle greetings that arrive DURING an active conversation."""
    global evi_last_activity, request_start_time
    global audio_chunk_count, total_audio_bytes, first_audio_time

    while True:
        try:
            greeting = None
            try:
                greeting = _greeting_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            if not greeting and os.path.exists(GREETING_FILE):
                with open(GREETING_FILE, 'r') as f:
                    greeting = f.read().strip()
                if greeting:
                    os.remove(GREETING_FILE)

            if greeting:
                now = time.time()
                words = greeting.split()
                name = words[1].rstrip('!') if len(words) > 1 else "someone"

                if name in _last_greeted and (now - _last_greeted[name]) < GREETING_COOLDOWN:
                    log(f"(skip {name} - recent)")
                else:
                    _last_greeted[name] = now
                    log(f"FACE (mid-convo): {greeting}")
                    audio_chunk_count = 0
                    total_audio_bytes = 0
                    first_audio_time = 0.0
                    evi_last_activity = now
                    request_start_time = time.time()
                    await socket.send_publish(UserInput(text=f"[Greet briefly]: {greeting}"))

        except Exception as e:
            log(f"Greeting error: {e}")

        await asyncio.sleep(0.5)


async def conversation_watchdog():
    """Cancel tasks when conversation goes idle."""
    global evi_last_activity
    while True:
        await asyncio.sleep(2.0)
        if evi_last_activity > 0 and (time.time() - evi_last_activity) > CONVERSATION_TIMEOUT:
            log("Conversation idle, disconnecting")
            raise asyncio.CancelledError


async def run_conversation(initial_greeting):
    """Connect to Hume, have conversation, disconnect on timeout."""
    global evi_last_activity, in_conversation, request_start_time
    global audio_chunk_count, total_audio_bytes, first_audio_time

    client = AsyncHumeClient(api_key=HUME_API_KEY)
    stream = Stream.new()
    mic = PulseAudioMic(source="ec_source", rate=16000, channels=1)

    connect_kwargs = {}
    if HUME_CONFIG_ID:
        connect_kwargs["config_id"] = HUME_CONFIG_ID

    log("Connecting to Hume...")

    async with client.empathic_voice.chat.connect(**connect_kwargs) as socket:
        log("Connected!")

        # Send audio config - interrupt=false until AEC is fixed
        audio_config = AudioConfiguration(sample_rate=16000, channels=1, encoding="linear16")
        await socket.send_publish(message=SessionSettings(
            audio=audio_config,
            allow_user_interrupt=False
        ))
        log("Audio config sent (interrupt=false until AEC fixed)")

        if LED_AVAILABLE:
            get_led().listening()

        # Send the triggering greeting
        in_conversation = True
        evi_last_activity = time.time()
        request_start_time = time.time()
        audio_chunk_count = 0
        total_audio_bytes = 0
        first_audio_time = 0.0

        log(f"FACE: {initial_greeting}")
        await socket.send_publish(UserInput(text=f"[Greet briefly]: {initial_greeting}"))

        async def handle_messages():
            async for message in socket:
                await on_message(message, stream)

        tasks = asyncio.gather(
            handle_messages(),
            mic.start_capture(socket),
            play_audio(stream),
            check_greetings_during_conversation(socket),
            conversation_watchdog(),
        )

        try:
            await tasks
        except asyncio.CancelledError:
            pass

    in_conversation = False
    if LED_AVAILABLE:
        get_led().off()
    log("Disconnected")


async def main() -> None:
    setup_ipc()
    log("Number 5 is alive! (on-demand Hume connection)")

    while True:
        log("Waiting for face detection...")
        greeting = await wait_for_greeting()

        log(f"Trigger received: {greeting}")
        try:
            await run_conversation(greeting)
        except Exception as e:
            log(f"Session error: {e}")

        log("Session ended, back to waiting")


if __name__ == "__main__":
    asyncio.run(main())
