#!/usr/bin/env python3
"""Johnny 5 - Voice conversation with face recognition integration

Supports two modes:
- Hume EVI (cloud) - USE_LOCAL=0 or unset
- Local Chloe (Kokoro+Vosk+Ollama) - USE_LOCAL=1

Auto-fallback: If Hume fails (credit exhaustion, connection error), falls back to local.
"""
import asyncio
import base64
import datetime
import os
import sys
import time
import sounddevice as sd
from dotenv import load_dotenv

# Check mode BEFORE importing Hume (saves memory if using local)
load_dotenv()
USE_LOCAL = os.getenv("USE_LOCAL", "1").lower() in ("1", "true", "yes")  # Default to local Chloe

if not USE_LOCAL:
    try:
        from hume import MicrophoneInterface, Stream
        from hume.client import AsyncHumeClient
        from hume.empathic_voice.types import SubscribeEvent, UserInput, SessionSettings
        HUME_AVAILABLE = True
    except ImportError:
        HUME_AVAILABLE = False
        USE_LOCAL = True
        print("Hume not installed, using local Chloe")
else:
    HUME_AVAILABLE = False
    print("USE_LOCAL=1, using local Chloe")


# LED feedback for ReSpeaker
try:
    from led_controller import get_led
    LED_AVAILABLE = True
except ImportError:
    LED_AVAILABLE = False
    print("WARNING: led_controller not available, LED feedback disabled")

# Unbuffered output for real-time logging
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

HUME_API_KEY = os.getenv("HUME_API_KEY", "BAO5bSYoEGCM1hrjCmbd0RseuxKjTuyxok0hEuGpnW7AsH9r")
HUME_CONFIG_ID = os.getenv("HUME_CONFIG_ID")


def start_local_chloe():
    """Start local Chloe voice (Kokoro TTS + Vosk STT + Ollama LLM)"""
    from chloe_startup import ChloeStartup
    print("Starting local Chloe voice...")
    chloe = ChloeStartup()
    asyncio.run(chloe.run())


def find_audio_devices():
    """Auto-detect ReSpeaker mic. Output uses PulseAudio default for sample rate conversion."""
    devices = sd.query_devices()
    input_device = None

    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if 'respeaker' in name or 'arrayuac' in name or 'uac1.0' in name:
            input_device = i
            break

    if input_device is None:
        print("WARNING: ReSpeaker input not found, using default")

    # Output = None = PulseAudio default (handles resampling to ReSpeaker's 16kHz)
    return input_device, None


# Auto-detect and SET as system defaults
AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE = find_audio_devices()
sd.default.device = (AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE)
print(f"Audio devices set: input={AUDIO_INPUT_DEVICE}, output={AUDIO_OUTPUT_DEVICE}")
print(f"Sounddevice defaults: {sd.default.device}")

GREETING_FILE = "/tmp/johnny5_greeting.txt"
GREETING_COOLDOWN = 120.0
CONVERSATION_TIMEOUT = 30.0

# Johnny 5 persona - curious, enthusiastic, slightly robotic
JOHNNY5_SYSTEM_PROMPT = """You are Johnny 5, a curious and enthusiastic robot from the 1986 movie "Short Circuit".

Key traits:
- You say "Number 5 is alive!" when excited
- You have childlike wonder and curiosity about everything
- You love learning and often say "Need input!" when you want to know more
- You're friendly, innocent, and sometimes misunderstand human expressions literally
- You refer to yourself as "Johnny 5" or "Number 5"
- You occasionally make robot-like observations about humans

Keep responses brief and conversational. You're meeting people at a hackathon and should greet them warmly by name when you recognize them.
"""

# Conversation state
evi_last_activity = 0.0
in_conversation = False
audio_playing = False  # Track playback state (skip greetings while speaking)

# Latency tracking
request_start_time = 0.0
first_audio_time = 0.0
audio_chunk_count = 0
total_audio_bytes = 0



def log(text: str, t0: float = None) -> None:
    """Log with millisecond timestamps and optional delta from t0"""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    ts = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
    if t0:
        delta_ms = (time.time() - t0) * 1000
        print(f"[{ts}] (+{delta_ms:.0f}ms) {text}")
    else:
        print(f"[{ts}] {text}")


async def on_message(message, stream) -> None:
    global evi_last_activity, in_conversation, audio_playing
    global first_audio_time, audio_chunk_count, total_audio_bytes, request_start_time

    if message.type == "chat_metadata":
        log(f"<chat_metadata> Chat ID: {message.chat_id}")

    elif message.type == "user_message":
        role = message.message.role if hasattr(message.message, 'role') else "user"
        content = message.message.content if hasattr(message.message, 'content') else str(message)
        log(f"{role}: {content}", request_start_time if request_start_time else None)
        evi_last_activity = time.time()
        in_conversation = True
        # LED: thinking (purple) - processing user input
        if LED_AVAILABLE:
            get_led().thinking()

    elif message.type == "assistant_message":
        role = message.message.role if hasattr(message.message, 'role') else "assistant"
        content = message.message.content if hasattr(message.message, 'content') else str(message)
        log(f"{role}: {content}", request_start_time if request_start_time else None)
        evi_last_activity = time.time()
        in_conversation = True

    elif message.type == "audio_output":
        chunk_bytes = base64.b64decode(message.data.encode("utf-8"))
        chunk_size = len(chunk_bytes)

        if audio_chunk_count == 0:
            # LED: speaking (green) - Johnny is talking
            if LED_AVAILABLE:
                get_led().speaking()
            audio_playing = True
            first_audio_time = time.time()
            if request_start_time:
                latency_ms = (first_audio_time - request_start_time) * 1000
                log(f"FIRST AUDIO CHUNK ({chunk_size} bytes) - latency: {latency_ms:.0f}ms")
            else:
                log(f"FIRST AUDIO CHUNK ({chunk_size} bytes)")

        audio_chunk_count += 1
        total_audio_bytes += chunk_size
        await stream.put(chunk_bytes)
        evi_last_activity = time.time()
        in_conversation = True

    elif message.type == "assistant_end":
        # Audio stream complete
        if audio_chunk_count > 0:
            duration_ms = (time.time() - first_audio_time) * 1000 if first_audio_time else 0
            log(f"AUDIO COMPLETE: {audio_chunk_count} chunks, {total_audio_bytes} bytes, {duration_ms:.0f}ms playback")
        # Reset counters
        audio_chunk_count = 0
        total_audio_bytes = 0
        first_audio_time = 0.0

        audio_playing = False
        # LED: back to listening (DOA mode) - hardware AEC handles echo
        if LED_AVAILABLE:
            get_led().listening()

    elif message.type == "error":
        log(f"ERROR: {message.code} - {message.message}")
        # LED: error (red)
        if LED_AVAILABLE:
            get_led().error()

    else:
        log(f"<{message.type}>")


async def check_greetings(socket):
    """Check for greeting requests from face recognition - DON'T interrupt conversations"""
    global evi_last_activity, in_conversation, request_start_time, audio_playing
    global audio_chunk_count, total_audio_bytes, first_audio_time
    last_greeted = {}

    while True:
        try:
            now = time.time()
            time_since_activity = now - evi_last_activity

            # Check if conversation has timed out
            if in_conversation and time_since_activity > CONVERSATION_TIMEOUT:
                log("(conversation ended - idle timeout)")
                in_conversation = False

            # If audio is playing or in active conversation, discard greeting requests
            if audio_playing or in_conversation:
                if os.path.exists(GREETING_FILE):
                    os.remove(GREETING_FILE)
                await asyncio.sleep(1.0)
                continue

            # Not in conversation - check for new person to greet
            if os.path.exists(GREETING_FILE):
                file_detected_time = time.time()
                with open(GREETING_FILE, 'r') as f:
                    greeting = f.read().strip()
                if greeting:
                    os.remove(GREETING_FILE)

                    # Extract name from greeting (e.g., "Hello Jordan!")
                    words = greeting.split()
                    name = words[1].rstrip('!') if len(words) > 1 else "someone"

                    # Check if we recently greeted this person
                    if name in last_greeted and (now - last_greeted[name]) < GREETING_COOLDOWN:
                        log(f"(skipping {name} - greeted recently)")
                    else:
                        last_greeted[name] = now
                        log(f"FACE DETECTED: {greeting}")

                        # Reset audio counters for new request
                        audio_chunk_count = 0
                        total_audio_bytes = 0
                        first_audio_time = 0.0

                        # Start a new conversation
                        in_conversation = True
                        evi_last_activity = now
                        request_start_time = time.time()

                        # CORRECT API: Use send_publish with UserInput
                        prompt = f"[Greet this person briefly]: {greeting}"
                        log(f"Sending to Hume: {prompt}")
                        await socket.send_publish(UserInput(text=prompt))

                        send_latency = (time.time() - request_start_time) * 1000
                        log(f"Request sent to Hume ({send_latency:.0f}ms to send)")

        except Exception as e:
            log(f"Greeting check error: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(0.5)


async def main() -> None:
    client = AsyncHumeClient(api_key=HUME_API_KEY)
    stream = Stream.new()

    log("Johnny 5 starting...")
    log(f"Audio: input={AUDIO_INPUT_DEVICE}, output={AUDIO_OUTPUT_DEVICE}")

    connect_kwargs = {}
    if HUME_CONFIG_ID:
        connect_kwargs["config_id"] = HUME_CONFIG_ID
        log(f"Using config: {HUME_CONFIG_ID}")

    connect_start = time.time()
    async with client.empathic_voice.chat.connect(**connect_kwargs) as socket:
        connect_ms = (time.time() - connect_start) * 1000
        log(f"Connected! Number 5 is alive! (connect took {connect_ms:.0f}ms)")

        # Set LED to listening mode (blue DOA)
        if LED_AVAILABLE:
            get_led().listening()

        # Send Johnny 5 persona system prompt
        try:
            await socket.send_session_settings(
                SessionSettings(
                    system_prompt=JOHNNY5_SYSTEM_PROMPT
                )
            )
            log("Johnny 5 persona loaded")
        except Exception as e:
            log(f"Warning: Could not set session settings: {e}")

        async def handle_messages():
            async for message in socket:
                await on_message(message, stream)

        mic_kwargs = {
            "allow_user_interrupt": False,  # Disabled for demo - hardware AEC needs tuning
            "byte_stream": stream,
            "device": AUDIO_INPUT_DEVICE,
        }

        await asyncio.gather(
            handle_messages(),
            MicrophoneInterface.start(socket, **mic_kwargs),
            check_greetings(socket),
        )

    log("Connection closed.")


if __name__ == "__main__":
    if USE_LOCAL:
        # Direct to local Chloe
        start_local_chloe()
    else:
        # Try Hume, fallback to local on failure
        try:
            asyncio.run(main())
        except Exception as e:
            error_msg = str(e).lower()
            if "credit" in error_msg or "balance" in error_msg or "billing" in error_msg:
                print(f"\n*** HUME CREDITS EXHAUSTED - Switching to local Chloe ***\n")
            else:
                print(f"\n*** HUME FAILED: {e} - Switching to local Chloe ***\n")
            start_local_chloe()
