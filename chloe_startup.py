#!/usr/bin/env python3
"""Chloe Startup - Local voice with Kokoro TTS + Qwen LLM

This is the LOCAL voice stack for when Hume EVI is unavailable.
Communicates with whoami_full.py via file-based IPC:
  - Reads: /tmp/johnny5_greeting.txt (face recognition greetings)
  - Writes: /tmp/johnny5_enroll.txt (enrollment requests)

First person enrolled becomes admin. Admin can make others admin.
"""

# Force Kokoro TTS to CPU - leave GPU for Ollama LLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import asyncio
import logging
import pickle
import time
import re
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# File IPC paths (same as whoami_full.py)
GREETING_FILE = "/tmp/johnny5_greeting.txt"
ENROLL_FILE = "/tmp/johnny5_enroll.txt"
FACE_DB_PATH = os.path.expanduser("~/whoami/face_database.pkl")

# Import TTS and LLM
try:
    from tts_kokoro import KokoroTTS
    HAS_KOKORO = True
except ImportError:
    HAS_KOKORO = False
    logger.warning("Kokoro TTS not available")

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    logger.warning("Ollama not available")

# LED feedback
try:
    from led_controller import get_led
    HAS_LED = True
except ImportError:
    HAS_LED = False

# STT
try:
    from vosk import Model, KaldiRecognizer
    import sounddevice as sd
    import queue
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False
    logger.warning("Vosk STT not available")


class ChloeVoice:
    """Chloe's voice using local Kokoro TTS + Qwen LLM"""

    SYSTEM_PROMPT = """You are Chloe, a friendly social robot. You are the Johnny5 robot but chose the name Chloe.

Key traits:
- Warm, curious, and genuinely interested in people
- You remember everyone you meet (face and voice recognition)
- Brief responses - 1-2 sentences max in conversation
- When someone tells you their name, you should enroll their face

Commands you understand:
- If someone says their name is X, call enroll_face to remember them
- If asked "who am I", describe what you see/hear
- If asked "make X admin", only do it if speaker is current admin

Keep responses natural and friendly. You're meeting people at a hackathon."""

    def __init__(self):
        self.tts = KokoroTTS(voice="af_heart") if HAS_KOKORO else None
        self.history = []
        self.admin = None
        self.current_speaker = None
        self.audio_queue = queue.Queue() if HAS_VOSK else None
        self.load_admin()

    def load_admin(self):
        """Load admin from face database"""
        if os.path.exists(FACE_DB_PATH):
            try:
                with open(FACE_DB_PATH, "rb") as f:
                    data = pickle.load(f)
                self.admin = data.get("admin")
                logger.info(f"Admin loaded: {self.admin}")
            except Exception as e:
                logger.error(f"Failed to load admin: {e}")

    def has_admin(self) -> bool:
        """Check if admin exists in face database"""
        self.load_admin()
        return self.admin is not None

    def speak(self, text: str):
        """Speak text using Kokoro TTS"""
        if not text:
            return
        logger.info(f"Speaking: {text}")
        if HAS_LED:
            get_led().speaking()
        if self.tts:
            self.tts.say(text)
        if HAS_LED:
            get_led().listening()

    def think(self, user_input: str) -> str:
        """Generate response using Qwen via Ollama"""
        if not HAS_OLLAMA:
            return "I cannot think right now, my brain is offline."

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(self.history[-10:])  # Keep last 10 turns
        messages.append({"role": "user", "content": user_input})

        try:
            if HAS_LED:
                get_led().thinking()
            response = ollama.chat(
                model="qwen2.5:1.5b",
                messages=messages,
                stream=False
            )
            reply = response["message"]["content"]

            # Update history
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": reply})

            # Check for enrollment intent
            self._check_enrollment_intent(user_input, reply)

            return reply
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I had trouble thinking about that."

    def _check_enrollment_intent(self, user_input: str, reply: str):
        """Check if user is introducing themselves and trigger enrollment"""
        intro_patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"it's (\w+)",  # Response to "what's your name"
        ]

        lower_input = user_input.lower()
        for pattern in intro_patterns:
            match = re.search(pattern, lower_input)
            if match:
                name = match.group(1).title()
                self.enroll_face(name)
                break

    def enroll_face(self, name: str):
        """Request face enrollment by writing to IPC file"""
        try:
            with open(ENROLL_FILE, "w") as f:
                f.write(name)
            logger.info(f"Enrollment requested for: {name}")
        except Exception as e:
            logger.error(f"Enrollment request failed: {e}")

    def make_admin(self, name: str, requester: str) -> str:
        """Make someone admin (only if requester is current admin)"""
        if not self.admin:
            return f"No admin exists yet. {name} should enroll first."
        if requester.lower() != self.admin.lower():
            return f"Sorry, only {self.admin} can make others admin."

        # Update face database
        try:
            with open(FACE_DB_PATH, "rb") as f:
                data = pickle.load(f)
            data["admin"] = name
            with open(FACE_DB_PATH, "wb") as f:
                pickle.dump(data, f)
            self.admin = name
            return f"{name} is now an admin!"
        except Exception as e:
            return f"Failed to update admin: {e}"


class ChloeStartup:
    """Handles Chloe's startup sequence and main loop"""

    def __init__(self):
        self.voice = ChloeVoice()
        self.running = True
        self.in_conversation = False
        self.last_greeting_time = 0
        self.greeting_cooldown = 60.0

    def report_status(self) -> str:
        """Generate status report"""
        status_parts = ["Systems online."]

        # Check face database
        if os.path.exists(FACE_DB_PATH):
            try:
                with open(FACE_DB_PATH, "rb") as f:
                    data = pickle.load(f)
                names = data.get("names", [])
                unique_names = list(set(names))
                admin = data.get("admin")
                status_parts.append(f"I know {len(unique_names)} {'person' if len(unique_names) == 1 else 'people'}.")
                if admin:
                    status_parts.append(f"My admin is {admin}.")
            except:
                pass
        else:
            status_parts.append("No face database found.")

        # Check components
        components = []
        if HAS_KOKORO:
            components.append("voice")
        if HAS_OLLAMA:
            components.append("brain")
        if HAS_VOSK:
            components.append("ears")
        if HAS_LED:
            components.append("lights")

        if components:
            status_parts.append(f"Active: {', '.join(components)}.")

        return " ".join(status_parts)

    async def startup_sequence(self):
        """Run startup sequence"""
        logger.info("=== Chloe Startup ===")

        # Set LED to startup mode
        if HAS_LED:
            get_led().listening()
            await asyncio.sleep(0.5)

        # Report status
        status = self.report_status()
        self.voice.speak(f"Hello! I'm Chloe. {status}")

        # Check for admin
        if not self.voice.has_admin():
            await asyncio.sleep(1.0)
            self.voice.speak("I don't have an admin yet. The first person I meet will become my admin. Please look at the camera and tell me your name.")

        if HAS_LED:
            get_led().listening()

    async def check_greetings(self):
        """Check for greetings from face recognition"""
        while self.running:
            try:
                if os.path.exists(GREETING_FILE):
                    now = time.time()

                    # Read greeting
                    with open(GREETING_FILE, "r") as f:
                        greeting = f.read().strip()
                    os.remove(GREETING_FILE)

                    if greeting:
                        # Check cooldown
                        if now - self.last_greeting_time < self.greeting_cooldown:
                            logger.info(f"Skipping greeting (cooldown): {greeting}")
                        else:
                            self.last_greeting_time = now
                            logger.info(f"Greeting from face recognition: {greeting}")

                            # Respond to greeting
                            response = self.voice.think(f"[Face recognition says]: {greeting}")
                            self.voice.speak(response)

            except Exception as e:
                logger.error(f"Greeting check error: {e}")

            await asyncio.sleep(0.5)

    async def voice_loop(self):
        """Main voice interaction loop using Vosk STT"""
        if not HAS_VOSK:
            logger.warning("Vosk not available, voice loop disabled")
            return

        # Use sounddevice default (already set by johnny5.py if called from there)
        input_device = sd.default.device[0] if sd.default.device[0] is not None else None
        logger.info(f"Using audio input device: {input_device}")

        # Load Vosk model
        model_path = os.path.expanduser("~/vosk-model-small-en-us-0.15")
        if not os.path.exists(model_path):
            model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            logger.error("Vosk model not found")
            return

        model = Model(model_path)
        recognizer = KaldiRecognizer(model, 16000)

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            audio_queue.put(bytes(indata))

        logger.info("Starting voice recognition...")

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            device=input_device,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            while self.running:
                try:
                    data = audio_queue.get(timeout=0.5)
                    if recognizer.AcceptWaveform(data):
                        import json
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()

                        if text and len(text) > 2:
                            logger.info(f"Heard: {text}")

                            # Generate and speak response
                            response = self.voice.think(text)
                            self.voice.speak(response)

                except queue.Empty:
                    pass
                except Exception as e:
                    logger.error(f"Voice loop error: {e}")

                await asyncio.sleep(0.01)

    async def run(self):
        """Main run loop"""
        await self.startup_sequence()

        # Run greeting checker and voice loop concurrently
        tasks = [
            asyncio.create_task(self.check_greetings()),
        ]

        if HAS_VOSK:
            tasks.append(asyncio.create_task(self.voice_loop()))
        else:
            # Without STT, just wait for greetings
            logger.info("Voice input disabled (no Vosk). Waiting for face recognition greetings.")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.running = False


async def main():
    """Entry point"""
    chloe = ChloeStartup()
    try:
        await chloe.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        chloe.running = False


if __name__ == "__main__":
    asyncio.run(main())
