#!/usr/bin/env python3
"""Johnny 5 - Modular voice conversation with automatic failover.

Uses the voice factory pattern for:
- Hume EVI (tethered/cloud) - primary
- Local Chloe (Kokoro+Vosk+Ollama) - fallback
- Mock (testing) - last resort

Communication via IPC message bus (replaces file-based /tmp/*.txt).
"""

import asyncio
import logging
import os
import sys
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

# Load environment
load_dotenv()

# Import voice factory and IPC
from voice import VoiceType, create_voice_stack, VoiceEvent
from voice.factory import FactoryConfig, VoiceFactory, FailoverEvent
from ipc import get_bus, Topic, VoiceChannel, VisionChannel, ActuatorChannel

# LED feedback (optional)
try:
    from led_controller import get_led
    LED_AVAILABLE = True
except ImportError:
    LED_AVAILABLE = False
    logger.warning("LED controller not available")


# =============================================================================
# Configuration
# =============================================================================

GREETING_COOLDOWN = 120.0  # Seconds before re-greeting same person
CONVERSATION_TIMEOUT = 30.0  # Seconds of idle before conversation ends

# Johnny 5 persona
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


# =============================================================================
# Conversation State
# =============================================================================

class ConversationState:
    """Track conversation state across the system."""

    def __init__(self):
        self.in_conversation = False
        self.last_activity = 0.0
        self.audio_playing = False
        self.last_greeted = {}  # name -> timestamp

    def start_conversation(self):
        self.in_conversation = True
        self.last_activity = time.time()

    def end_conversation(self):
        self.in_conversation = False

    def update_activity(self):
        self.last_activity = time.time()

    def is_idle(self, timeout: float = CONVERSATION_TIMEOUT) -> bool:
        return time.time() - self.last_activity > timeout

    def can_greet(self, name: str, cooldown: float = GREETING_COOLDOWN) -> bool:
        last = self.last_greeted.get(name, 0)
        return time.time() - last > cooldown

    def mark_greeted(self, name: str):
        self.last_greeted[name] = time.time()


state = ConversationState()


# =============================================================================
# IPC Handlers
# =============================================================================

def setup_ipc_handlers():
    """Connect IPC channels to conversation handlers."""
    bus = get_bus()
    vision = VisionChannel(bus, source="johnny5")
    voice = VoiceChannel(bus, source="johnny5")
    actuator = ActuatorChannel(bus, source="johnny5")

    # Handle greetings from face recognition
    def on_greeting(msg):
        text = msg.data.get("text", "")
        if not text or state.audio_playing or state.in_conversation:
            return

        # Extract name from greeting
        words = text.split()
        name = words[1].rstrip("!") if len(words) > 1 else "someone"

        if not state.can_greet(name):
            logger.info(f"Skipping {name} - greeted recently")
            return

        state.mark_greeted(name)
        state.start_conversation()
        logger.info(f"Face greeting: {text}")

        # Emit to voice system
        voice.publish_transcript(f"[Greet this person briefly]: {text}")

    bus.subscribe(Topic.VISION_GREETING, on_greeting)

    # Handle enrollment requests from voice
    def on_enroll_request(msg):
        name = msg.data.get("name", "")
        if name:
            logger.info(f"Enrollment request: {name}")
            # Forward to vision system
            vision.request_enrollment(name)

    bus.subscribe(Topic.VISION_ENROLL_REQUEST, on_enroll_request)

    # LED state updates
    def on_speaking_start(msg):
        state.audio_playing = True
        state.update_activity()
        if LED_AVAILABLE:
            get_led().speaking()

    def on_speaking_stop(msg):
        state.audio_playing = False
        if LED_AVAILABLE:
            get_led().listening()

    def on_listening_start(msg):
        if LED_AVAILABLE:
            get_led().listening()

    bus.subscribe(Topic.VOICE_SPEAKING_START, on_speaking_start)
    bus.subscribe(Topic.VOICE_SPEAKING_STOP, on_speaking_stop)
    bus.subscribe(Topic.VOICE_LISTENING_START, on_listening_start)

    return bus, vision, voice, actuator


# =============================================================================
# Legacy file-based IPC bridge (for backward compatibility)
# =============================================================================

GREETING_FILE = "/tmp/johnny5_greeting.txt"
ENROLL_FILE = "/tmp/johnny5_enroll.txt"


async def file_ipc_bridge(vision: VisionChannel):
    """Bridge legacy file-based IPC to message bus.

    This allows whoami_full.py to work without modification.
    TODO: Update whoami_full.py to use IPC directly and remove this.
    """
    while True:
        try:
            # Check for greeting file
            if os.path.exists(GREETING_FILE):
                with open(GREETING_FILE, "r") as f:
                    greeting = f.read().strip()
                os.remove(GREETING_FILE)
                if greeting:
                    vision.publish_greeting(greeting)

            # Check for enrollment file
            if os.path.exists(ENROLL_FILE):
                with open(ENROLL_FILE, "r") as f:
                    name = f.read().strip()
                os.remove(ENROLL_FILE)
                if name:
                    vision.request_enrollment(name)

        except Exception as e:
            logger.error(f"File IPC bridge error: {e}")

        await asyncio.sleep(0.5)


# =============================================================================
# Main Application
# =============================================================================

class Johnny5:
    """Main Johnny 5 application using modular voice factory."""

    def __init__(self):
        self.factory = None
        self.stack = None
        self.running = False

    async def setup(self):
        """Initialize voice factory and IPC."""
        # Create factory config
        config = FactoryConfig(
            hume_api_key=os.getenv("HUME_API_KEY"),
            hume_config_id=os.getenv("HUME_CONFIG_ID"),
            system_prompt=JOHNNY5_SYSTEM_PROMPT,
            llm_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            auto_failover=True,
        )

        # Create factory with failover logging
        self.factory = VoiceFactory(config)

        def on_failover(event: FailoverEvent):
            logger.warning(
                f"Voice failover: {event.from_backend} -> {event.to_backend} "
                f"({event.reason.value}: {event.error_message})"
            )
            if LED_AVAILABLE:
                get_led().error()
                time.sleep(0.5)

        self.factory.on_failover(on_failover)

        # Setup IPC
        bus, vision, voice, actuator = setup_ipc_handlers()
        self.vision = vision
        self.voice = voice
        self.actuator = actuator

        # Determine voice type from environment
        # Default: AUTO (tries Hume first, falls back to local if it fails)
        # Set USE_LOCAL=1 to force local Chloe mode
        use_local = os.getenv("USE_LOCAL", "0").lower() in ("1", "true", "yes")
        voice_type = VoiceType.LOCAL if use_local else VoiceType.HUME

        logger.info(f"Creating voice stack (type={voice_type.value})...")

        # Create voice stack (with automatic failover)
        self.stack = await self.factory.create(voice_type)

        status = self.stack.status()
        logger.info(f"Voice stack ready: {status}")

        # Wire up voice events to IPC
        self.stack.on(VoiceEvent.TRANSCRIPT, lambda msg: self.voice.publish_transcript(msg.text))
        self.stack.on(VoiceEvent.SPEAKING_START, lambda msg: self.voice.publish_speaking_start())
        self.stack.on(VoiceEvent.SPEAKING_STOP, lambda msg: self.voice.publish_speaking_stop())

    async def run(self):
        """Main run loop."""
        self.running = True

        logger.info("Johnny 5 starting...")
        if LED_AVAILABLE:
            get_led().listening()

        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self.stack.conversation_loop()),
            asyncio.create_task(file_ipc_bridge(self.vision)),
            asyncio.create_task(self._idle_checker()),
        ]

        logger.info("Number 5 is alive!")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            await self.stack.stop()

    async def _idle_checker(self):
        """Check for conversation timeout."""
        while self.running:
            if state.in_conversation and state.is_idle():
                logger.info("Conversation ended (idle timeout)")
                state.end_conversation()
            await asyncio.sleep(1.0)


async def main():
    """Entry point."""
    johnny = Johnny5()

    try:
        await johnny.setup()
        await johnny.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
