#!/usr/bin/env python3
"""Speaker Fusion - Unified voice+face identification for Hume EVI.

Integrates:
- DOA (direction of arrival) from ReSpeaker mic array
- Spatial detection from OAK-D camera
- Face recognition from WhoAmI/DeepFace
- Voice recognition from WeSpeaker
- IPC bus for real-time updates to EVI

This gives Hume EVI "eyes" - it can ask:
- "Who is speaking?" (voice + DOA + face fusion)
- "Where are they?" (3D position from spatial tracker)
- "Who else is here?" (all visible people)

Usage:
    from speaker_fusion import get_speaker_fusion, SpeakerFusion

    fusion = get_speaker_fusion()
    await fusion.start()

    # Get current speaker with full identity
    speaker = fusion.get_current_speaker()
    if speaker:
        print(f"{speaker['name']} speaking from {speaker['direction']}°")
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from doa_spatial_fusion import DOASpatialFusion, SpeakerMatch, get_fusion
    DOA_FUSION_AVAILABLE = True
except ImportError:
    DOA_FUSION_AVAILABLE = False
    logger.warning("DOA spatial fusion not available")

try:
    from ipc import get_bus, Topic, Message
    IPC_AVAILABLE = True
except ImportError:
    IPC_AVAILABLE = False
    logger.warning("IPC not available")


@dataclass
class Speaker:
    """Unified speaker information combining all modalities."""
    # Identity
    name: Optional[str] = None
    is_known: bool = False

    # Confidence scores
    face_confidence: float = 0.0
    voice_confidence: float = 0.0
    fusion_confidence: float = 0.0

    # Spatial info
    direction: float = 0.0  # DOA angle in degrees
    distance: float = 0.0   # Distance in mm
    position: Optional[Tuple[float, float, float]] = None  # (x, y, z) mm

    # Tracking
    track_id: Optional[int] = None
    is_speaking: bool = False
    last_seen: float = field(default_factory=time.time)

    # Sources used for identification
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for IPC/tool responses."""
        return {
            "name": self.name,
            "is_known": self.is_known,
            "confidence": self.fusion_confidence,
            "face_confidence": self.face_confidence,
            "voice_confidence": self.voice_confidence,
            "direction": self.direction,
            "distance": self.distance,
            "position": self.position,
            "track_id": self.track_id,
            "is_speaking": self.is_speaking,
            "sources": self.sources,
        }


class SpeakerFusion:
    """Unified speaker identification combining DOA, face, and voice.

    Provides a single source of truth for "who is speaking" by fusing:
    - DOA angle from microphone array
    - Face recognition from camera
    - Voice recognition from audio

    Publishes updates to IPC bus for Hume EVI tool calls.
    """

    def __init__(self):
        self._doa_fusion: Optional[DOASpatialFusion] = None
        self._bus = None
        self._running = False

        # Current state
        self._current_speaker: Optional[Speaker] = None
        self._visible_people: Dict[str, Speaker] = {}
        self._lock = threading.Lock()

        # Identity providers (set externally)
        self._face_identifier: Optional[Callable] = None  # bbox -> identity
        self._voice_identifier: Optional[Callable] = None  # audio -> identity

        # Voice embedding cache (speaker name -> embedding)
        self._voice_embeddings: Dict[str, Any] = {}

        # Callbacks
        self._on_speaker_change: List[Callable[[Speaker], None]] = []

    def set_face_identifier(self, callback: Callable[[Any], Optional[Dict]]):
        """Set the face identification callback.

        Callback signature: def identify(detection) -> {'name': str, 'confidence': float}
        """
        self._face_identifier = callback

    def set_voice_identifier(self, callback: Callable[[bytes], Optional[Dict]]):
        """Set the voice identification callback.

        Callback signature: def identify(audio_chunk) -> {'name': str, 'confidence': float}
        """
        self._voice_identifier = callback

    def on_speaker_change(self, callback: Callable[[Speaker], None]):
        """Register callback for speaker changes."""
        self._on_speaker_change.append(callback)

    async def start(self) -> bool:
        """Start the speaker fusion system."""
        try:
            # Initialize IPC
            if IPC_AVAILABLE:
                self._bus = get_bus()
                self._subscribe_ipc()
                logger.info("Speaker fusion: IPC connected")

            # Initialize DOA fusion
            if DOA_FUSION_AVAILABLE:
                self._doa_fusion = get_fusion()
                self._doa_fusion.set_identity_callback(self._identify_from_detection)
                self._doa_fusion.on_speaker_change(self._on_doa_speaker_change)

                if self._doa_fusion.start():
                    logger.info("Speaker fusion: DOA fusion started")
                else:
                    logger.warning("Speaker fusion: DOA fusion failed to start")

            self._running = True
            logger.info("Speaker fusion: Started")
            return True

        except Exception as e:
            logger.error(f"Speaker fusion: Failed to start: {e}")
            return False

    async def stop(self):
        """Stop the speaker fusion system."""
        self._running = False
        if self._doa_fusion:
            self._doa_fusion.stop()
        logger.info("Speaker fusion: Stopped")

    def _subscribe_ipc(self):
        """Subscribe to IPC topics for identity updates."""
        if not self._bus:
            return

        # Face recognition updates
        self._bus.subscribe(Topic.VISION_FACE_RECOGNIZED, self._on_face_recognized)

        # Voice recognition updates
        self._bus.subscribe(Topic.AUDIO_SPEAKER_IDENTIFIED, self._on_voice_recognized)

        # DOA updates (for when fusion isn't available)
        self._bus.subscribe(Topic.AUDIO_DOA, self._on_doa_update)

    def _on_face_recognized(self, msg: Message):
        """Handle face recognition from IPC."""
        name = msg.data.get("name")
        confidence = msg.data.get("confidence", 0.0)
        bbox = msg.data.get("bbox")
        track_id = msg.data.get("track_id")

        if not name:
            return

        with self._lock:
            # Update visible people cache
            if name not in self._visible_people:
                self._visible_people[name] = Speaker(name=name, is_known=True)

            speaker = self._visible_people[name]
            speaker.face_confidence = confidence
            speaker.track_id = track_id
            speaker.last_seen = time.time()
            if "face" not in speaker.sources:
                speaker.sources.append("face")

            # If this is the current speaker, update their info
            if self._current_speaker and self._current_speaker.name == name:
                self._current_speaker.face_confidence = confidence

    def _on_voice_recognized(self, msg: Message):
        """Handle voice recognition from IPC."""
        name = msg.data.get("name")
        confidence = msg.data.get("confidence", 0.0)

        if not name:
            return

        with self._lock:
            if name not in self._visible_people:
                self._visible_people[name] = Speaker(name=name, is_known=True)

            speaker = self._visible_people[name]
            speaker.voice_confidence = confidence
            if "voice" not in speaker.sources:
                speaker.sources.append("voice")

            # Update current speaker if they're speaking
            if self._current_speaker and self._current_speaker.name == name:
                self._current_speaker.voice_confidence = confidence

    def _on_doa_update(self, msg: Message):
        """Handle raw DOA update from IPC (backup if fusion unavailable)."""
        direction = msg.data.get("direction", 0)
        is_speaking = msg.data.get("is_speaking", False)

        with self._lock:
            if self._current_speaker:
                self._current_speaker.direction = direction
                self._current_speaker.is_speaking = is_speaking

    def _on_doa_speaker_change(self, match: SpeakerMatch):
        """Handle speaker change from DOA fusion."""
        with self._lock:
            if not match.is_valid:
                # No valid speaker
                if self._current_speaker and self._current_speaker.is_speaking:
                    self._current_speaker.is_speaking = False
                    self._publish_speaker_update()
                return

            # Build speaker from fusion match
            speaker = Speaker(
                direction=match.doa_angle,
                is_speaking=True,
                fusion_confidence=match.fusion_score,
            )

            if match.detection:
                speaker.distance = match.detection.distance
                speaker.position = (match.detection.x, match.detection.y, match.detection.z)
                speaker.track_id = getattr(match.detection, 'track_id', None)

                # Try to identify from face
                identity = self._identify_from_detection(match.detection)
                if identity:
                    speaker.name = identity.get("name")
                    speaker.is_known = True
                    speaker.face_confidence = identity.get("confidence", 0.0)
                    speaker.sources.append("face")

            # Check if this is a different speaker
            changed = (
                self._current_speaker is None or
                self._current_speaker.name != speaker.name or
                abs(self._current_speaker.direction - speaker.direction) > 20
            )

            self._current_speaker = speaker

            if changed:
                self._publish_speaker_update()
                self._notify_callbacks(speaker)

    def _identify_from_detection(self, detection) -> Optional[Dict]:
        """Identify a person from their spatial detection."""
        if self._face_identifier:
            try:
                return self._face_identifier(detection)
            except Exception as e:
                logger.error(f"Face identification error: {e}")
        return None

    def _publish_speaker_update(self):
        """Publish speaker update to IPC."""
        if not self._bus or not self._current_speaker:
            return

        self._bus.emit(
            Topic.AUDIO_SPEAKER_IDENTIFIED,
            source="speaker_fusion",
            **self._current_speaker.to_dict()
        )

    def _notify_callbacks(self, speaker: Speaker):
        """Notify registered callbacks of speaker change."""
        for cb in self._on_speaker_change:
            try:
                cb(speaker)
            except Exception as e:
                logger.error(f"Speaker callback error: {e}")

    # =========================================================================
    # Public API - Used by recognition handler and tools
    # =========================================================================

    def get_current_speaker(self) -> Optional[Dict]:
        """Get the current speaker with full identity info.

        Returns:
            Speaker dict or None if no one speaking
        """
        with self._lock:
            if self._current_speaker and self._current_speaker.is_speaking:
                return self._current_speaker.to_dict()
        return None

    def get_speaker_direction(self) -> Dict[str, Any]:
        """Get direction of current speaker (for look_at_speaker tool).

        Returns:
            {'direction': float, 'confidence': float, 'is_speaking': bool}
        """
        with self._lock:
            if self._current_speaker:
                return {
                    "direction": self._current_speaker.direction,
                    "confidence": self._current_speaker.fusion_confidence,
                    "is_speaking": self._current_speaker.is_speaking,
                    "name": self._current_speaker.name,
                }
        return {
            "direction": None,
            "confidence": 0.0,
            "is_speaking": False,
        }

    def get_visible_people(self) -> List[Dict]:
        """Get all visible people with their recognition status.

        Returns:
            List of speaker dicts
        """
        with self._lock:
            # Clean up stale entries (not seen in 5 seconds)
            cutoff = time.time() - 5.0
            self._visible_people = {
                k: v for k, v in self._visible_people.items()
                if v.last_seen > cutoff
            }

            return [s.to_dict() for s in self._visible_people.values()]

    def identify_speaker(self) -> Dict[str, Any]:
        """Identify who is currently speaking using all modalities.

        Returns:
            {'name': str, 'confidence': float, 'sources': list} or None
        """
        with self._lock:
            if not self._current_speaker:
                return {"name": None, "is_known": False}

            return {
                "name": self._current_speaker.name,
                "is_known": self._current_speaker.is_known,
                "confidence": self._current_speaker.fusion_confidence,
                "face_confidence": self._current_speaker.face_confidence,
                "voice_confidence": self._current_speaker.voice_confidence,
                "direction": self._current_speaker.direction,
                "sources": self._current_speaker.sources,
            }

    def who_said_that(self, lookback_seconds: float = 5.0) -> Dict[str, Any]:
        """Identify who just spoke (for who_said_that tool).

        Args:
            lookback_seconds: How far back to check

        Returns:
            Speaker identification result
        """
        # For now, return current speaker - could add history tracking
        return self.identify_speaker()

    async def enroll_speaker_voice(self, name: str, duration: float = 10.0) -> Dict:
        """Enroll a speaker's voice for recognition.

        Args:
            name: Name of person to enroll
            duration: How long to record

        Returns:
            {'success': bool, 'message': str}
        """
        # Publish enrollment request
        if self._bus:
            self._bus.emit(
                Topic.AUDIO_ENROLL_VOICE,
                source="speaker_fusion",
                name=name,
                duration=duration,
            )

        return {
            "success": True,
            "name": name,
            "message": f"Recording {name}'s voice for {duration} seconds...",
        }


# =============================================================================
# Singleton
# =============================================================================
_speaker_fusion: Optional[SpeakerFusion] = None
_fusion_lock = threading.Lock()


def get_speaker_fusion() -> SpeakerFusion:
    """Get the singleton speaker fusion instance."""
    global _speaker_fusion
    with _fusion_lock:
        if _speaker_fusion is None:
            _speaker_fusion = SpeakerFusion()
        return _speaker_fusion


# =============================================================================
# Integration with Recognition Handler
# =============================================================================

async def create_fusion_handler():
    """Create a recognition handler that uses speaker fusion.

    Returns a handler function compatible with ToolExecutionEngine.
    """
    fusion = get_speaker_fusion()
    await fusion.start()

    async def handler(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recognition tool calls using speaker fusion."""

        if tool_name == "get_speaker_direction":
            return fusion.get_speaker_direction()

        elif tool_name == "identify_speaker":
            return fusion.identify_speaker()

        elif tool_name == "get_visible_people":
            return {
                "people": fusion.get_visible_people(),
                "count": len(fusion.get_visible_people()),
            }

        elif tool_name == "who_said_that":
            lookback = args.get("lookback_seconds", 5.0)
            return fusion.who_said_that(lookback)

        elif tool_name == "enroll_voice":
            name = args.get("name")
            duration = args.get("duration_seconds", 10.0)
            if not name:
                return {"success": False, "error": "Name required"}
            return await fusion.enroll_speaker_voice(name, duration)

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    return handler


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    import asyncio

    async def main():
        print("Speaker Fusion Test")
        print("=" * 50)
        print("Speak to see unified identification...")
        print("Press Ctrl+C to exit\n")

        fusion = get_speaker_fusion()

        def on_speaker(speaker: Speaker):
            if speaker.name:
                print(f"\n[SPEAKER] {speaker.name} at {speaker.direction:.0f}°")
                print(f"  Face: {speaker.face_confidence:.0%} | Voice: {speaker.voice_confidence:.0%}")
            else:
                print(f"\n[UNKNOWN SPEAKER] at {speaker.direction:.0f}°")

        fusion.on_speaker_change(on_speaker)
        await fusion.start()

        try:
            while True:
                speaker = fusion.get_current_speaker()
                if speaker and speaker.get("is_speaking"):
                    name = speaker.get("name", "Unknown")
                    direction = speaker.get("direction", 0)
                    conf = speaker.get("confidence", 0)
                    print(f"{name} speaking @ {direction:.0f}° (conf: {conf:.0%})  ", end='\r')
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
            await fusion.stop()
            print("Test complete!")

    asyncio.run(main())
