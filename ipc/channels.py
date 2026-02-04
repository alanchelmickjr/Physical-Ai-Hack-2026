"""Pre-defined channels for common robot subsystems.

These provide typed interfaces over the message bus,
making it easier to publish/subscribe to specific data.
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional, List

from .bus import MessageBus, Message, Topic, get_bus


@dataclass
class PersonDetection:
    """Person detected by vision system."""
    track_id: int
    name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    centroid: tuple  # (cx, cy)
    is_speaking: bool = False


@dataclass
class DOAReading:
    """Direction of arrival from microphone array."""
    angle: float  # degrees, 0 = forward
    confidence: float
    is_speaking: bool


class VoiceChannel:
    """Channel for voice events (transcripts, responses, state)."""

    def __init__(self, bus: Optional[MessageBus] = None, source: str = "voice"):
        self._bus = bus or get_bus()
        self._source = source

    def on_transcript(self, handler: Callable[[str], None]) -> None:
        """Subscribe to user transcripts."""
        def wrapper(msg: Message):
            handler(msg.data.get("text", ""))
        self._bus.subscribe(Topic.VOICE_TRANSCRIPT, wrapper)

    def on_response(self, handler: Callable[[str], None]) -> None:
        """Subscribe to LLM responses."""
        def wrapper(msg: Message):
            handler(msg.data.get("text", ""))
        self._bus.subscribe(Topic.VOICE_RESPONSE, wrapper)

    def on_speaking_start(self, handler: Callable[[], None]) -> None:
        """Subscribe to speaking start events."""
        def wrapper(msg: Message):
            handler()
        self._bus.subscribe(Topic.VOICE_SPEAKING_START, wrapper)

    def on_speaking_stop(self, handler: Callable[[], None]) -> None:
        """Subscribe to speaking stop events."""
        def wrapper(msg: Message):
            handler()
        self._bus.subscribe(Topic.VOICE_SPEAKING_STOP, wrapper)

    def publish_transcript(self, text: str) -> None:
        """Publish a user transcript."""
        self._bus.emit(Topic.VOICE_TRANSCRIPT, source=self._source, text=text)

    def publish_response(self, text: str) -> None:
        """Publish an LLM response."""
        self._bus.emit(Topic.VOICE_RESPONSE, source=self._source, text=text)

    def publish_speaking_start(self) -> None:
        """Signal that robot started speaking."""
        self._bus.emit(Topic.VOICE_SPEAKING_START, source=self._source)

    def publish_speaking_stop(self) -> None:
        """Signal that robot stopped speaking."""
        self._bus.emit(Topic.VOICE_SPEAKING_STOP, source=self._source)


class VisionChannel:
    """Channel for vision events (face recognition, tracking)."""

    def __init__(self, bus: Optional[MessageBus] = None, source: str = "vision"):
        self._bus = bus or get_bus()
        self._source = source

    def on_face_recognized(self, handler: Callable[[str, float], None]) -> None:
        """Subscribe to face recognition events. Handler receives (name, confidence)."""
        def wrapper(msg: Message):
            handler(msg.data.get("name", ""), msg.data.get("confidence", 0.0))
        self._bus.subscribe(Topic.VISION_FACE_RECOGNIZED, wrapper)

    def on_greeting(self, handler: Callable[[str], None]) -> None:
        """Subscribe to greeting messages from face recognition."""
        def wrapper(msg: Message):
            handler(msg.data.get("text", ""))
        self._bus.subscribe(Topic.VISION_GREETING, wrapper)

    def on_enroll_request(self, handler: Callable[[str], None]) -> None:
        """Subscribe to enrollment requests. Handler receives name to enroll."""
        def wrapper(msg: Message):
            handler(msg.data.get("name", ""))
        self._bus.subscribe(Topic.VISION_ENROLL_REQUEST, wrapper)

    def publish_face_recognized(self, name: str, confidence: float, track_id: int = 0) -> None:
        """Publish face recognition result."""
        self._bus.emit(
            Topic.VISION_FACE_RECOGNIZED,
            source=self._source,
            name=name,
            confidence=confidence,
            track_id=track_id,
        )

    def publish_greeting(self, text: str) -> None:
        """Publish greeting message (replaces /tmp/johnny5_greeting.txt)."""
        self._bus.emit(Topic.VISION_GREETING, source=self._source, text=text)

    def request_enrollment(self, name: str) -> None:
        """Request face enrollment (replaces /tmp/johnny5_enroll.txt)."""
        self._bus.emit(Topic.VISION_ENROLL_REQUEST, source=self._source, name=name)

    def publish_enroll_complete(self, name: str, is_admin: bool = False) -> None:
        """Signal enrollment complete."""
        self._bus.emit(
            Topic.VISION_ENROLL_COMPLETE,
            source=self._source,
            name=name,
            is_admin=is_admin,
        )


class AudioChannel:
    """Channel for audio events (DOA, speaker ID, VAD)."""

    def __init__(self, bus: Optional[MessageBus] = None, source: str = "audio"):
        self._bus = bus or get_bus()
        self._source = source

    def on_doa(self, handler: Callable[[float, bool], None]) -> None:
        """Subscribe to DOA updates. Handler receives (angle, is_speaking)."""
        def wrapper(msg: Message):
            handler(msg.data.get("angle", 0.0), msg.data.get("is_speaking", False))
        self._bus.subscribe(Topic.AUDIO_DOA, wrapper)

    def on_speaker_id(self, handler: Callable[[str, float], None]) -> None:
        """Subscribe to speaker identification. Handler receives (name, confidence)."""
        def wrapper(msg: Message):
            handler(msg.data.get("name", ""), msg.data.get("confidence", 0.0))
        self._bus.subscribe(Topic.AUDIO_SPEAKER_ID, wrapper)

    def on_vad(self, handler: Callable[[bool], None]) -> None:
        """Subscribe to voice activity detection."""
        def wrapper(msg: Message):
            handler(msg.data.get("active", False))
        self._bus.subscribe(Topic.AUDIO_VAD, wrapper)

    def publish_doa(self, angle: float, is_speaking: bool = False, confidence: float = 1.0) -> None:
        """Publish DOA reading."""
        self._bus.emit(
            Topic.AUDIO_DOA,
            source=self._source,
            angle=angle,
            is_speaking=is_speaking,
            confidence=confidence,
        )

    def publish_speaker_id(self, name: str, confidence: float) -> None:
        """Publish speaker identification."""
        self._bus.emit(
            Topic.AUDIO_SPEAKER_ID,
            source=self._source,
            name=name,
            confidence=confidence,
        )

    def publish_vad(self, active: bool) -> None:
        """Publish voice activity detection."""
        self._bus.emit(Topic.AUDIO_VAD, source=self._source, active=active)


class SensorChannel:
    """Channel for sensor events (smell, touch, proximity, LIDAR)."""

    def __init__(self, bus: Optional[MessageBus] = None, source: str = "sensor"):
        self._bus = bus or get_bus()
        self._source = source

    def on_proximity(self, handler: Callable[[float, str], None]) -> None:
        """Subscribe to proximity events. Handler receives (distance, direction)."""
        def wrapper(msg: Message):
            handler(msg.data.get("distance", 0.0), msg.data.get("direction", ""))
        self._bus.subscribe(Topic.SENSOR_PROXIMITY, wrapper)

    def publish_proximity(self, distance: float, direction: str = "front") -> None:
        """Publish proximity reading (from LIDAR or ultrasonic)."""
        self._bus.emit(
            Topic.SENSOR_PROXIMITY,
            source=self._source,
            distance=distance,
            direction=direction,
        )

    def publish_smell(self, compound: str, intensity: float) -> None:
        """Publish smell sensor reading."""
        self._bus.emit(
            Topic.SENSOR_SMELL,
            source=self._source,
            compound=compound,
            intensity=intensity,
        )

    def publish_touch(self, location: str, pressure: float = 1.0) -> None:
        """Publish touch sensor reading."""
        self._bus.emit(
            Topic.SENSOR_TOUCH,
            source=self._source,
            location=location,
            pressure=pressure,
        )


class ActuatorChannel:
    """Channel for actuator commands (LED, motors, gantry)."""

    def __init__(self, bus: Optional[MessageBus] = None, source: str = "actuator"):
        self._bus = bus or get_bus()
        self._source = source

    def on_led_command(self, handler: Callable[[str], None]) -> None:
        """Subscribe to LED commands. Handler receives mode string."""
        def wrapper(msg: Message):
            handler(msg.data.get("mode", ""))
        self._bus.subscribe(Topic.ACTUATOR_LED, wrapper)

    def on_motor_command(self, handler: Callable[[str, float], None]) -> None:
        """Subscribe to motor commands. Handler receives (motor_id, value)."""
        def wrapper(msg: Message):
            handler(msg.data.get("motor_id", ""), msg.data.get("value", 0.0))
        self._bus.subscribe(Topic.ACTUATOR_MOTOR, wrapper)

    def command_led(self, mode: str) -> None:
        """Send LED command (listening, speaking, thinking, error)."""
        self._bus.emit(Topic.ACTUATOR_LED, source=self._source, mode=mode)

    def command_motor(self, motor_id: str, value: float) -> None:
        """Send motor command."""
        self._bus.emit(
            Topic.ACTUATOR_MOTOR,
            source=self._source,
            motor_id=motor_id,
            value=value,
        )

    def command_gantry(self, pan: float, tilt: float) -> None:
        """Send gantry (head) position command."""
        self._bus.emit(
            Topic.ACTUATOR_GANTRY,
            source=self._source,
            pan=pan,
            tilt=tilt,
        )
