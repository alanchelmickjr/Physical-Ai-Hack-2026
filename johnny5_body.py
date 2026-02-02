#!/usr/bin/env python3
"""Johnny Five Body Interface - High-Level Motor API

This is the top-level interface for controlling a Johnny Five robot's body.
It's designed for integration with the voice system (johnny5.py).

NOTE: "Johnny Five" is the robot MODEL.
      Each unit chooses its own name (e.g., "Chloe").

The robot thinks in human terms, not motor coordinates:
    "Look at who's talking"
    "Wave hello"
    "I can hear someone behind me"

Usage:
    from chloe_body import Johnny5Body

    body = Johnny5Body()
    await body.start()

    # Will automatically track speakers
    # For manual control:
    await body.wave()
    await body.look_at("speaker")
    await body.express("happy")
"""

import asyncio
from typing import Optional, Callable
from enum import Enum, auto

from motion_coordinator import MotionCoordinator, get_motion_coordinator
from head_tracker import SpeakerState, SpeakerLocation


class Emotion(Enum):
    """Emotions Chloe can express through body language."""
    HAPPY = auto()
    SAD = auto()
    EXCITED = auto()
    CURIOUS = auto()
    CONFUSED = auto()
    GREETING = auto()
    FAREWELL = auto()
    THINKING = auto()
    NEUTRAL = auto()


class Johnny5Body:
    """High-level interface to a Johnny Five robot's body.

    This class provides the abstraction layer between
    the robot's "mind" (voice AI) and "body" (motors).

    Key features:
    - Automatic speaker tracking (DOA → head movement)
    - Natural gestures triggered by conversation
    - Out-of-view awareness (knows someone is behind her)
    - Smooth, human-like motion
    """

    def __init__(self):
        self._coordinator: Optional[MotionCoordinator] = None
        self._is_speaking = False
        self._running = False

        # Callbacks for voice system
        self._on_out_of_view: Optional[Callable[[float, str], None]] = None

    async def start(self):
        """Start Chloe's body systems.

        This begins:
        - Speaker tracking (head follows sound)
        - Motion coordinator
        """
        if self._running:
            return

        self._coordinator = get_motion_coordinator()

        # Register callback for out-of-view speakers
        self._coordinator.head_tracker.on_out_of_range(self._handle_out_of_range)

        await self._coordinator.start()
        self._running = True
        print("Johnny5 Body: Ready")

    async def stop(self):
        """Stop all body systems and disable motors."""
        if not self._running:
            return

        if self._coordinator:
            await self._coordinator.stop()

        self._running = False
        print("Johnny5 Body: Stopped")

    # =========================================================================
    # Voice System Integration
    # =========================================================================

    def on_out_of_view_speaker(self, callback: Callable[[float, str], None]):
        """Register callback when someone speaks from outside view.

        This allows the robot to say "I can hear you behind me" or
        "There's someone to my left I can't see".

        callback(angle, description):
            angle: DOA in degrees
            description: "behind", "far left", "far right"
        """
        self._on_out_of_view = callback

    def _handle_out_of_range(self, doa: float, location: SpeakerLocation):
        """Internal handler for out-of-range speakers."""
        if self._on_out_of_view:
            descriptions = {
                SpeakerLocation.NEED_BASE_TURN: "to the side",
                SpeakerLocation.BEHIND: "behind",
            }
            desc = descriptions.get(location, "outside my view")

            # Make it more specific
            if doa > 90 and doa < 180:
                desc = "behind me to the right"
            elif doa >= 180 and doa < 270:
                desc = "behind me to the left"
            elif doa < -90 or doa > 270:
                desc = "behind me"

            self._on_out_of_view(doa, desc)

    def set_speaking(self, is_speaking: bool):
        """Tell the robot if it's currently speaking.

        When speaking, head tracking is less responsive
        (don't whip around mid-sentence).
        """
        self._is_speaking = is_speaking
        if self._coordinator:
            self._coordinator.set_speaking(is_speaking)

    # =========================================================================
    # High-Level Actions - What Chloe "thinks"
    # =========================================================================

    async def look_at(self, target: str):
        """Look at something.

        Args:
            target: "speaker", "forward", "left", "right", "up", "down"
                   or a person's name
        """
        if not self._coordinator:
            return

        if target == "speaker":
            await self._coordinator.look_at_speaker()
        elif target in ("forward", "left", "right", "up", "down"):
            await self._coordinator.look_direction(target)
        else:
            # Assume it's a person's name
            await self._coordinator.look_at_person(target)

    async def turn_to(self, target: str):
        """Turn the whole body toward something.

        Used when target is outside head range.

        Args:
            target: "speaker", "around" (180°), angle (degrees)
        """
        if not self._coordinator:
            return

        if target == "speaker":
            await self._coordinator.turn_to_speaker()
        elif target == "around":
            await self._coordinator.turn_around()
        else:
            try:
                angle = float(target)
                await self._coordinator._rotate_base(angle)
            except ValueError:
                print(f"Unknown turn target: {target}")

    async def wave(self, arm: str = "right"):
        """Wave hello or goodbye."""
        if self._coordinator:
            await self._coordinator.wave(arm)

    async def point(self, direction: str = "forward"):
        """Point in a direction."""
        if self._coordinator:
            await self._coordinator.point(direction)

    async def shrug(self):
        """Shrug (I don't know)."""
        if self._coordinator:
            await self._coordinator.shrug()

    async def express(self, emotion: str):
        """Express an emotion through body language.

        Args:
            emotion: "happy", "sad", "excited", "curious",
                    "confused", "greeting", "farewell", "thinking"
        """
        if not self._coordinator:
            return

        emotion = emotion.lower()

        if emotion in ("happy", "excited"):
            await self._coordinator.express_excitement()
        elif emotion == "thinking":
            await self._coordinator.express_thinking()
        elif emotion == "greeting":
            await self._coordinator.wave()
        elif emotion == "farewell":
            await self._coordinator.wave()
        elif emotion == "confused":
            await self._coordinator.shrug()
        # Other emotions: subtle gestures or no action

    async def home(self):
        """Return to neutral home position."""
        if self._coordinator:
            await self._coordinator.go_home()

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_speaker_location(self) -> Optional[dict]:
        """Get current speaker location info.

        Returns:
            {
                "angle": float,       # DOA in degrees
                "in_view": bool,      # Can camera see them?
                "direction": str,     # "left", "right", "center", "behind"
            }
        """
        if not self._coordinator:
            return None

        state = self._coordinator.head_tracker.get_state()
        if not state.is_speaking:
            return None

        # Determine direction description
        doa = state.doa
        if -45 <= doa <= 45:
            direction = "center"
        elif 45 < doa <= 135:
            direction = "right"
        elif -135 <= doa < -45:
            direction = "left"
        else:
            direction = "behind"

        return {
            "angle": doa,
            "in_view": state.location in (
                SpeakerLocation.IN_VIEW,
                SpeakerLocation.GANTRY_REACHABLE
            ),
            "direction": direction,
        }

    @property
    def is_tracking(self) -> bool:
        """Is head tracking active?"""
        return self._running


# Singleton
_body: Optional[Johnny5Body] = None


def get_johnny5_body() -> Johnny5Body:
    """Get the singleton Johnny5Body instance."""
    global _body
    if _body is None:
        _body = Johnny5Body()
    return _body


# Alias for convenience
get_body = get_johnny5_body


# Test
if __name__ == "__main__":
    async def test():
        print("Testing Johnny Five Body")
        print("=" * 50)

        body = get_johnny5_body()

        def on_out_of_view(angle: float, desc: str):
            print(f"\n  !! Someone is speaking {desc} ({angle:.0f}°)")

        body.on_out_of_view_speaker(on_out_of_view)

        await body.start()

        print("\nSpeak to test head tracking...")
        print("Commands: wave, point, shrug, think, express <emotion>, look <dir>, turn, home, quit")

        try:
            while True:
                cmd = input("\n> ").strip().lower()

                if cmd == "quit":
                    break
                elif cmd == "wave":
                    await body.wave()
                elif cmd == "point":
                    await body.point()
                elif cmd == "shrug":
                    await body.shrug()
                elif cmd == "think":
                    await body.express("thinking")
                elif cmd.startswith("express "):
                    emotion = cmd.split()[1]
                    await body.express(emotion)
                elif cmd.startswith("look "):
                    target = cmd.split()[1]
                    await body.look_at(target)
                elif cmd == "turn":
                    await body.turn_to("speaker")
                elif cmd == "home":
                    await body.home()
                elif cmd == "status":
                    loc = body.get_speaker_location()
                    print(f"Speaker: {loc}")
                else:
                    print("Unknown command")

        except KeyboardInterrupt:
            pass

        await body.stop()
        print("\nTest complete!")

    asyncio.run(test())
