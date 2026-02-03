"""Message bus for inter-component communication.

Provides pub/sub messaging between robot subsystems.
Thread-safe, async-compatible, with optional persistence.
"""

import asyncio
import json
import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class Topic(Enum):
    """Message topics for pub/sub routing."""

    # Voice events
    VOICE_TRANSCRIPT = "voice.transcript"        # User speech transcribed
    VOICE_RESPONSE = "voice.response"            # LLM response
    VOICE_SPEAKING_START = "voice.speaking.start"
    VOICE_SPEAKING_STOP = "voice.speaking.stop"
    VOICE_LISTENING_START = "voice.listening.start"
    VOICE_LISTENING_STOP = "voice.listening.stop"

    # Vision events
    VISION_PERSON_DETECTED = "vision.person.detected"
    VISION_PERSON_LOST = "vision.person.lost"
    VISION_FACE_RECOGNIZED = "vision.face.recognized"
    VISION_FACE_UNKNOWN = "vision.face.unknown"
    VISION_GREETING = "vision.greeting"          # Face recognition greeting
    VISION_ENROLL_REQUEST = "vision.enroll.request"  # Request face enrollment
    VISION_ENROLL_COMPLETE = "vision.enroll.complete"
    VISION_REQUEST_FACES = "vision.request.faces"  # Request current faces
    VISION_SNAPSHOT = "vision.snapshot"          # Request camera snapshot for Hume

    # Audio events
    AUDIO_DOA = "audio.doa"                      # Direction of arrival
    AUDIO_SPEAKER_ID = "audio.speaker.id"        # Speaker identified
    AUDIO_SPEAKER_IDENTIFIED = "audio.speaker.identified"  # Speaker name confirmed
    AUDIO_VAD = "audio.vad"                      # Voice activity detected
    AUDIO_REQUEST_DOA = "audio.request.doa"      # Request current DOA
    AUDIO_ENROLL_VOICE = "audio.enroll.voice"    # Start voice enrollment

    # Identity management events
    IDENTITY_UPDATED = "identity.updated"        # Person identity updated
    IDENTITY_FORGET = "identity.forget"          # Request to forget person
    IDENTITY_MERGE = "identity.merge"            # Merge two identities

    # Sensor events
    SENSOR_SMELL = "sensor.smell"
    SENSOR_TOUCH = "sensor.touch"
    SENSOR_PROXIMITY = "sensor.proximity"

    # Actuator commands
    ACTUATOR_LED = "actuator.led"
    ACTUATOR_MOTOR = "actuator.motor"
    ACTUATOR_GANTRY = "actuator.gantry"

    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_STATUS = "system.status"
    SYSTEM_SHUTDOWN = "system.shutdown"


@dataclass
class Message:
    """Message passed between components."""

    topic: Topic
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "topic": self.topic.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            topic=Topic(d["topic"]),
            data=d.get("data", {}),
            timestamp=d.get("timestamp", time.time()),
            source=d.get("source", ""),
            correlation_id=d.get("correlation_id"),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "Message":
        """Create from JSON string."""
        return cls.from_dict(json.loads(s))


# Type alias for handlers
MessageHandler = Callable[[Message], None]
AsyncMessageHandler = Callable[[Message], Any]  # Can be sync or async


class MessageBus:
    """Central message bus for component communication.

    Features:
    - Pub/sub pattern with topic filtering
    - Both sync and async handlers
    - Thread-safe for multi-threaded components
    - Optional message history for debugging
    - Wildcard subscriptions (e.g., "voice.*")
    """

    def __init__(self, history_size: int = 100):
        self._subscribers: Dict[Topic, List[AsyncMessageHandler]] = defaultdict(list)
        self._wildcard_subscribers: Dict[str, List[AsyncMessageHandler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._history: List[Message] = []
        self._history_size = history_size
        self._running = True

    def _get_async_lock(self) -> asyncio.Lock:
        """Get or create async lock."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def subscribe(self, topic: Topic, handler: AsyncMessageHandler) -> None:
        """Subscribe to a specific topic."""
        with self._lock:
            self._subscribers[topic].append(handler)
            logger.debug(f"Subscribed to {topic.value}: {handler}")

    def subscribe_pattern(self, pattern: str, handler: AsyncMessageHandler) -> None:
        """Subscribe to topics matching pattern (e.g., 'voice.*')."""
        with self._lock:
            self._wildcard_subscribers[pattern].append(handler)
            logger.debug(f"Subscribed to pattern {pattern}: {handler}")

    def unsubscribe(self, topic: Topic, handler: AsyncMessageHandler) -> None:
        """Unsubscribe from a topic."""
        with self._lock:
            if handler in self._subscribers[topic]:
                self._subscribers[topic].remove(handler)

    def publish(self, message: Message) -> None:
        """Publish a message (sync version)."""
        if not self._running:
            return

        # Add to history
        with self._lock:
            self._history.append(message)
            if len(self._history) > self._history_size:
                self._history.pop(0)

        # Get matching handlers
        handlers = self._get_handlers(message.topic)

        # Call handlers
        for handler in handlers:
            try:
                result = handler(message)
                # If it's a coroutine, schedule it
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop - run synchronously
                        asyncio.run(result)
            except Exception as e:
                logger.error(f"Handler error for {message.topic.value}: {e}")

    async def publish_async(self, message: Message) -> None:
        """Publish a message (async version)."""
        if not self._running:
            return

        # Add to history
        async with self._get_async_lock():
            self._history.append(message)
            if len(self._history) > self._history_size:
                self._history.pop(0)

        # Get matching handlers
        handlers = self._get_handlers(message.topic)

        # Call handlers concurrently
        tasks = []
        for handler in handlers:
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    tasks.append(asyncio.create_task(result))
            except Exception as e:
                logger.error(f"Handler error for {message.topic.value}: {e}")

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _get_handlers(self, topic: Topic) -> List[AsyncMessageHandler]:
        """Get all handlers matching a topic."""
        handlers = []

        with self._lock:
            # Exact match
            handlers.extend(self._subscribers.get(topic, []))

            # Wildcard matches
            topic_str = topic.value
            for pattern, pattern_handlers in self._wildcard_subscribers.items():
                if self._matches_pattern(topic_str, pattern):
                    handlers.extend(pattern_handlers)

        return handlers

    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports * wildcard)."""
        if pattern == "*":
            return True

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".")

        return topic == pattern

    def emit(self, topic: Topic, source: str = "", **data) -> None:
        """Convenience method to publish a message."""
        msg = Message(topic=topic, data=data, source=source)
        self.publish(msg)

    async def emit_async(self, topic: Topic, source: str = "", **data) -> None:
        """Async convenience method to publish a message."""
        msg = Message(topic=topic, data=data, source=source)
        await self.publish_async(msg)

    def get_history(self, topic: Optional[Topic] = None, limit: int = 10) -> List[Message]:
        """Get recent message history."""
        with self._lock:
            if topic:
                filtered = [m for m in self._history if m.topic == topic]
            else:
                filtered = list(self._history)
            return filtered[-limit:]

    def shutdown(self) -> None:
        """Shutdown the bus."""
        self._running = False
        self.emit(Topic.SYSTEM_SHUTDOWN, source="bus")


# Singleton instance
_bus: Optional[MessageBus] = None
_bus_lock = threading.Lock()


def get_bus() -> MessageBus:
    """Get the global message bus instance."""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = MessageBus()
    return _bus
