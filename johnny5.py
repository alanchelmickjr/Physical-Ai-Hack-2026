#!/usr/bin/env python3
"""Johnny 5 - Hume EVI voice conversation with face recognition integration

FIXES APPLIED:
- Correct API: send_publish(UserInput(...)) instead of send_user_input(text=...)
- Millisecond timestamps for latency tracking
- Audio chunk counting and timing
- Request/response correlation
"""
import asyncio
import base64
import datetime
import os
import sys
import time
import sounddevice as sd
from dotenv import load_dotenv
from hume import MicrophoneInterface, Stream
from hume.client import AsyncHumeClient
from hume.empathic_voice.types import SubscribeEvent, UserInput

# Unbuffered output for real-time logging
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

load_dotenv()

HUME_API_KEY = os.getenv("HUME_API_KEY", "BAO5bSYoEGCM1hrjCmbd0RseuxKjTuyxok0hEuGpnW7AsH9r")
HUME_CONFIG_ID = os.getenv("HUME_CONFIG_ID")


def find_audio_devices():
    """Auto-detect ReSpeaker mic and HDMI output by name."""
    devices = sd.query_devices()
    input_device = None
    output_device = None

    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if 'respeaker' in name and dev['max_input_channels'] > 0:
            input_device = i
        elif 'hdmi' in name and dev['max_output_channels'] > 0 and output_device is None:
            output_device = i

    if input_device is None:
        print("WARNING: ReSpeaker not found, using default input")
        input_device = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
    if output_device is None:
        print("WARNING: HDMI not found, using default output")
        output_device = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None

    return input_device, output_device


# Auto-detect and SET as system defaults
AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE = find_audio_devices()
sd.default.device = (AUDIO_INPUT_DEVICE, AUDIO_OUTPUT_DEVICE)
print(f"Audio devices set: input={AUDIO_INPUT_DEVICE}, output={AUDIO_OUTPUT_DEVICE}")
print(f"Sounddevice defaults: {sd.default.device}")

GREETING_FILE = "/tmp/johnny5_greeting.txt"
GREETING_COOLDOWN = 120.0
CONVERSATION_TIMEOUT = 30.0

# Conversation state
evi_last_activity = 0.0
in_conversation = False

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


async def on_message(message: SubscribeEvent, stream: Stream) -> None:
    global evi_last_activity, in_conversation
    global first_audio_time, audio_chunk_count, total_audio_bytes, request_start_time

    if message.type == "chat_metadata":
        log(f"<chat_metadata> Chat ID: {message.chat_id}")

    elif message.type == "user_message":
        role = message.message.role if hasattr(message.message, 'role') else "user"
        content = message.message.content if hasattr(message.message, 'content') else str(message)
        log(f"{role}: {content}", request_start_time if request_start_time else None)
        evi_last_activity = time.time()
        in_conversation = True

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

    elif message.type == "error":
        log(f"ERROR: {message.code} - {message.message}")

    else:
        log(f"<{message.type}>")


async def check_greetings(socket):
    """Check for greeting requests from face recognition - DON'T interrupt conversations"""
    global evi_last_activity, in_conversation, request_start_time
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

            # If in active conversation, just discard any greeting requests
            if in_conversation:
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

        async def handle_messages():
            async for message in socket:
                await on_message(message, stream)

        mic_kwargs = {
            "allow_user_interrupt": True,
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
    asyncio.run(main())
