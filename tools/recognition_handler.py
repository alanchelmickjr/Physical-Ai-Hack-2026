"""Recognition Handler - Bridges tool engine with face/voice recognition via IPC.

This module provides the handler that the ToolExecutionEngine uses to
query the recognition system for face identification, voice identification,
and related operations.

Integrates with SpeakerFusion for unified DOA + face + voice identification.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Optional speaker fusion integration
try:
    from speaker_fusion import get_speaker_fusion, SpeakerFusion
    SPEAKER_FUSION_AVAILABLE = True
except ImportError:
    SPEAKER_FUSION_AVAILABLE = False
    logger.info("Speaker fusion not available, using IPC-only mode")


@dataclass
class RecognitionConfig:
    """Configuration for recognition behavior."""
    # Tone down face recognition when we know who we're talking to
    reduce_when_identified: bool = True
    # How long to keep reduced rate after identification (seconds)
    reduced_rate_duration: float = 30.0
    # Normal face recognition interval (seconds)
    normal_interval: float = 1.0
    # Reduced face recognition interval (seconds)
    reduced_interval: float = 5.0
    # Voice recognition is always active
    voice_always_active: bool = True
    # Use speaker fusion when available (DOA + face + voice)
    use_speaker_fusion: bool = True


class RecognitionHandler:
    """Handles recognition queries from the tool engine.

    Interfaces with face recognition (WhoAmI/DeepFace) and voice
    recognition (WeSpeaker) via IPC channels.
    """

    def __init__(self, config: Optional[RecognitionConfig] = None):
        self.config = config or RecognitionConfig()
        self._bus = None
        self._vision_channel = None
        self._audio_channel = None

        # Speaker fusion (unified DOA + face + voice)
        self._speaker_fusion: Optional[Any] = None

        # Track current speaker for rate limiting
        self._current_speaker: Optional[str] = None
        self._speaker_identified_at: float = 0.0

        # Cache recent recognition results
        self._visible_people: Dict[str, Any] = {}
        self._speaker_direction: Optional[Dict[str, Any]] = None
        self._known_people: Dict[str, Any] = {}

        # IPC subscriptions
        self._subscriptions = []

    async def initialize(self) -> None:
        """Initialize IPC connections and speaker fusion."""
        # Try to use speaker fusion (unified DOA + face + voice)
        if self.config.use_speaker_fusion and SPEAKER_FUSION_AVAILABLE:
            try:
                self._speaker_fusion = get_speaker_fusion()
                await self._speaker_fusion.start()
                logger.info("Recognition handler: Using speaker fusion")
            except Exception as e:
                logger.warning(f"Speaker fusion init failed: {e}, falling back to IPC")
                self._speaker_fusion = None

        # Also initialize IPC for direct subscriptions
        try:
            from ipc import get_bus, Topic, VisionChannel, AudioChannel
            self._bus = get_bus()
            self._vision_channel = VisionChannel(self._bus)
            self._audio_channel = AudioChannel(self._bus)

            # Subscribe to recognition updates
            self._subscriptions.append(
                self._bus.subscribe(Topic.VISION_FACE_RECOGNIZED, self._on_face_recognized)
            )
            self._subscriptions.append(
                self._bus.subscribe(Topic.AUDIO_SPEAKER_IDENTIFIED, self._on_speaker_identified)
            )
            self._subscriptions.append(
                self._bus.subscribe(Topic.AUDIO_DOA, self._on_doa_update)
            )
            logger.info("Recognition handler: IPC connected")
        except ImportError:
            logger.warning("IPC not available")
            pass

    async def shutdown(self) -> None:
        """Cleanup IPC connections."""
        for unsub in self._subscriptions:
            unsub()
        self._subscriptions.clear()

    async def handle(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a recognition query from the tool engine.

        Args:
            tool_name: Name of the recognition tool
            args: Tool arguments

        Returns:
            Recognition result data
        """
        handlers = {
            "get_visible_people": self._handle_get_visible_people,
            "identify_person": self._handle_identify_person,
            "enroll_person": self._handle_enroll_person,
            "get_speaker_direction": self._handle_get_speaker_direction,
            "identify_speaker": self._handle_identify_speaker,
            "enroll_voice": self._handle_enroll_voice,
            "get_known_people": self._handle_get_known_people,
            "forget_person": self._handle_forget_person,
            "who_said_that": self._handle_who_said_that,
            "get_last_seen": self._handle_get_last_seen,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown recognition tool: {tool_name}"}

        return await handler(args)

    # =========================================================================
    # Vision Recognition Handlers
    # =========================================================================

    async def _handle_get_visible_people(self, args: Dict) -> Dict[str, Any]:
        """Get list of people currently visible to camera."""
        # Use speaker fusion if available (has unified view)
        if self._speaker_fusion:
            people = self._speaker_fusion.get_visible_people()
            return {
                "people": people,
                "count": len(people),
            }

        # Fallback to IPC-based approach
        if self._bus:
            from ipc import Topic
            # Publish request
            self._bus.publish(Topic.VISION_REQUEST_FACES, {})
            # Wait briefly for response
            await asyncio.sleep(0.1)

        # Return cached visible people
        people = []
        for name, data in self._visible_people.items():
            people.append({
                "name": name,
                "confidence": data.get("confidence", 0.0),
                "bbox": data.get("bbox"),
                "is_speaking": data.get("is_speaking", False),
            })

        return {
            "people": people,
            "count": len(people),
        }

    async def _handle_identify_person(self, args: Dict) -> Dict[str, Any]:
        """Identify a specific person by bbox or index."""
        index = args.get("index", 0)
        bbox = args.get("bbox")

        if bbox:
            # Find person at bbox
            for name, data in self._visible_people.items():
                if self._bbox_overlap(bbox, data.get("bbox", [])):
                    return {
                        "name": name,
                        "confidence": data.get("confidence", 0.0),
                        "known": True
                    }
        elif index < len(self._visible_people):
            people = list(self._visible_people.items())
            name, data = people[index]
            return {
                "name": name,
                "confidence": data.get("confidence", 0.0),
                "known": True
            }

        return {"name": None, "known": False, "message": "Person not found"}

    async def _handle_enroll_person(self, args: Dict) -> Dict[str, Any]:
        """Enroll a new person into the recognition database."""
        name = args.get("name")
        if not name:
            return {"success": False, "error": "Name required"}

        # Request enrollment via IPC
        if self._bus:
            from ipc import Topic
            self._bus.publish(Topic.VISION_ENROLL_REQUEST, {"name": name})

        return {
            "success": True,
            "name": name,
            "message": f"Enrollment requested for {name}"
        }

    # =========================================================================
    # Audio Recognition Handlers
    # =========================================================================

    async def _handle_get_speaker_direction(self, args: Dict) -> Dict[str, Any]:
        """Get direction of current speaker from DOA."""
        # Use speaker fusion (combines DOA + face + voice)
        if self._speaker_fusion:
            return self._speaker_fusion.get_speaker_direction()

        # Fallback to cached direction
        if self._speaker_direction:
            return self._speaker_direction

        # Request fresh DOA data via IPC
        if self._bus:
            from ipc import Topic
            self._bus.publish(Topic.AUDIO_REQUEST_DOA, {})
            await asyncio.sleep(0.05)

        if self._speaker_direction:
            return self._speaker_direction

        return {
            "direction": None,
            "confidence": 0.0,
            "is_speaking": False,
            "message": "No active speaker detected"
        }

    async def _handle_identify_speaker(self, args: Dict) -> Dict[str, Any]:
        """Identify the current speaker by voice."""
        # Use speaker fusion for unified identification
        if self._speaker_fusion:
            return self._speaker_fusion.identify_speaker()

        # Fallback to IPC-based approach
        if not self._current_speaker:
            return {
                "name": None,
                "known": False,
                "message": "No speaker identified"
            }

        return {
            "name": self._current_speaker,
            "known": True,
            "direction": self._speaker_direction.get("direction") if self._speaker_direction else None
        }

    async def _handle_enroll_voice(self, args: Dict) -> Dict[str, Any]:
        """Enroll a person's voice for speaker recognition."""
        name = args.get("name")
        if not name:
            return {"success": False, "error": "Name required"}

        duration = args.get("duration_seconds", 10.0)

        # Use speaker fusion if available
        if self._speaker_fusion:
            return await self._speaker_fusion.enroll_speaker_voice(name, duration)

        # Fallback to IPC
        if self._bus:
            from ipc import Topic
            self._bus.publish(Topic.AUDIO_ENROLL_VOICE, {"name": name, "duration": duration})

        return {
            "success": True,
            "name": name,
            "message": f"Voice enrollment started for {name}. Please speak for {duration} seconds."
        }

    async def _handle_who_said_that(self, args: Dict) -> Dict[str, Any]:
        """Identify who said the last thing (combines voice + DOA + face)."""
        # Use speaker fusion for unified identification
        if self._speaker_fusion:
            lookback = args.get("lookback_seconds", 5.0)
            return self._speaker_fusion.who_said_that(lookback)

        # Fallback to IPC-based approach
        result = {
            "speaker": None,
            "confidence": 0.0,
            "sources": []
        }

        # Voice identification
        if self._current_speaker:
            result["speaker"] = self._current_speaker
            result["confidence"] = 0.8
            result["sources"].append("voice")

        # DOA + face fusion
        if self._speaker_direction:
            direction = self._speaker_direction.get("direction")
            # Find face at that direction
            for name, data in self._visible_people.items():
                if data.get("is_speaking"):
                    if result["speaker"] == name:
                        result["confidence"] = min(0.95, result["confidence"] + 0.15)
                        result["sources"].append("face")
                    elif not result["speaker"]:
                        result["speaker"] = name
                        result["confidence"] = 0.7
                        result["sources"].append("face")

        return result

    # =========================================================================
    # Database Handlers
    # =========================================================================

    async def _handle_get_known_people(self, args: Dict) -> Dict[str, Any]:
        """Get list of all known people in the database."""
        # Query via IPC or direct database access
        return {
            "people": list(self._known_people.keys()),
            "count": len(self._known_people)
        }

    async def _handle_forget_person(self, args: Dict) -> Dict[str, Any]:
        """Remove a person from the recognition database."""
        name = args.get("name")
        if not name:
            return {"success": False, "error": "Name required"}

        # Request deletion via IPC
        if self._bus:
            from ipc import Topic
            self._bus.publish(Topic.IDENTITY_FORGET, {"name": name})

        if name in self._known_people:
            del self._known_people[name]

        return {
            "success": True,
            "name": name,
            "message": f"Removed {name} from database"
        }

    async def _handle_get_last_seen(self, args: Dict) -> Dict[str, Any]:
        """Get when a person was last seen."""
        name = args.get("name")
        if not name:
            return {"error": "Name required"}

        person = self._known_people.get(name)
        if not person:
            return {
                "name": name,
                "known": False,
                "message": f"{name} is not in the database"
            }

        return {
            "name": name,
            "known": True,
            "last_seen": person.get("last_seen"),
            "last_seen_ago": person.get("last_seen_ago")
        }

    # =========================================================================
    # IPC Callbacks
    # =========================================================================

    def _on_face_recognized(self, msg) -> None:
        """Handle face recognition update from IPC."""
        name = msg.data.get("name")
        if name:
            self._visible_people[name] = {
                "confidence": msg.data.get("confidence", 0.0),
                "bbox": msg.data.get("bbox"),
                "is_speaking": msg.data.get("is_speaking", False),
                "timestamp": asyncio.get_event_loop().time()
            }

            # Update known people cache
            if name not in self._known_people:
                self._known_people[name] = {}
            self._known_people[name]["last_seen"] = msg.data.get("timestamp")

    def _on_speaker_identified(self, msg) -> None:
        """Handle speaker identification update from IPC."""
        import time
        self._current_speaker = msg.data.get("name")
        self._speaker_identified_at = time.time()

    def _on_doa_update(self, msg) -> None:
        """Handle DOA (direction of arrival) update from IPC."""
        self._speaker_direction = {
            "direction": msg.data.get("direction"),
            "confidence": msg.data.get("confidence", 0.0),
            "is_speaking": msg.data.get("is_speaking", False),
        }

    # =========================================================================
    # Utilities
    # =========================================================================

    def _bbox_overlap(self, bbox1, bbox2) -> bool:
        """Check if two bounding boxes overlap significantly."""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return False

        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return False

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)

        return intersection / area1 > 0.5 if area1 > 0 else False

    def should_reduce_face_recognition(self) -> bool:
        """Check if face recognition should be reduced.

        Returns True if we've recently identified someone and should
        tone down recognition rate.
        """
        if not self.config.reduce_when_identified:
            return False

        import time
        elapsed = time.time() - self._speaker_identified_at
        return elapsed < self.config.reduced_rate_duration

    def get_recognition_interval(self) -> float:
        """Get the current face recognition interval based on state."""
        if self.should_reduce_face_recognition():
            return self.config.reduced_interval
        return self.config.normal_interval


def create_recognition_handler(config: Optional[RecognitionConfig] = None) -> RecognitionHandler:
    """Factory function to create a recognition handler."""
    return RecognitionHandler(config)
