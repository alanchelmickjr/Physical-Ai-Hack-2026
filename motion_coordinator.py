#!/usr/bin/env python3
"""Motion Coordinator - Johnny Five Motor Abstraction Layer

This is the "spine" that connects sensory input to motor output.
It translates high-level intents into coordinated motor actions.

Flow:
    Sensors (DOA, Vision) → Motion Coordinator → Motors (Solo-CLI)

Features:
- Head tracking: Look at who's talking
- Base rotation: Turn to face out-of-view speakers
- Arm gestures: Express emotion during conversation
- Attention system: Remember where people were

Human-like Abstraction:
    "Look at who's talking" → DOA → head tracking
    "Turn to see them" → out-of-range detection → base rotation
    "Wave hello" → gesture → arm motion
    "I can't see you" → speaker behind → turn around
"""

import asyncio
import subprocess
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
from enum import Enum, auto

from head_tracker import HeadTracker, SpeakerState, SpeakerLocation, get_head_tracker
from doa_reader import ReSpeakerDOA, get_doa


class Gesture(Enum):
    """Arm gestures Chloe can perform."""
    WAVE = auto()
    POINT = auto()
    SHRUG = auto()
    THINKING = auto()
    EXCITED = auto()
    CALM = auto()
    GREETING = auto()
    FAREWELL = auto()


class AttentionTarget(Enum):
    """What Chloe is paying attention to."""
    SPEAKER = auto()      # Whoever is talking
    PERSON = auto()       # A specific remembered person
    OBJECT = auto()       # Something she's looking at
    EXPLORATION = auto()  # Scanning the room
    REST = auto()         # Neutral position


@dataclass
class MotorConfig:
    """Hardware configuration for motors."""
    # Serial ports
    left_port: str = "/dev/ttyACM0"
    right_port: str = "/dev/ttyACM1"

    # Motor IDs
    left_arm_ids: tuple = (1, 2, 3, 4, 5, 6)
    right_arm_ids: tuple = (1, 2, 3, 4, 5, 6)
    wheel_ids: tuple = (7, 8, 9)    # On left port
    lift_id: int = 10                # On left port
    gantry_ids: tuple = (7, 8)      # Pan, Tilt on right port

    # Limits
    pan_min: float = -90.0
    pan_max: float = 90.0
    tilt_min: float = -45.0
    tilt_max: float = 45.0

    # Speeds (0-1)
    track_speed: float = 0.5
    gesture_speed: float = 0.7
    base_speed: float = 0.3


@dataclass
class PersonMemory:
    """Memory of where a person was last seen/heard."""
    name: Optional[str] = None
    last_doa: float = 0.0
    last_seen_time: float = 0.0
    is_current_speaker: bool = False
    confidence: float = 0.0


class MotionCoordinator:
    """Coordinates all of a Johnny Five robot's movements.

    This is the abstraction layer between:
    - Sensors (DOA, Vision, Identity)
    - Motors (Gantry, Arms, Base, Lift)

    The robot doesn't think in motor coordinates.
    It thinks: "look at who's talking", "wave hello", "turn around"
    This class translates those intents to motor commands.
    """

    def __init__(self, config: Optional[MotorConfig] = None):
        self.config = config or MotorConfig()

        # Subsystems
        self.doa = get_doa()
        self.head_tracker = get_head_tracker()

        # State
        self._attention = AttentionTarget.REST
        self._current_speaker: Optional[PersonMemory] = None
        self._known_people: Dict[str, PersonMemory] = {}
        self._is_speaking = False  # Is Chloe talking?

        # Motor state
        self._current_pan: float = 0.0
        self._current_tilt: float = 0.0
        self._base_angle: float = 0.0  # Estimated base orientation

        self._running = False
        self._lock = threading.Lock()

    async def start(self):
        """Start the motion coordinator."""
        if self._running:
            return

        # Connect head tracker to gantry
        self.head_tracker.set_gantry_controller(self._move_gantry_sync)

        # Handle out-of-range speakers
        self.head_tracker.on_out_of_range(self._on_out_of_range)

        # Handle speaker detection
        self.head_tracker.on_speaker_detected(self._on_speaker_detected)

        # Start head tracking
        self.head_tracker.start(poll_rate_hz=30)

        self._running = True
        print("Motion Coordinator: Started")

    async def stop(self):
        """Stop the motion coordinator."""
        self._running = False
        self.head_tracker.stop()
        await self._disable_motors()
        print("Motion Coordinator: Stopped")

    # =========================================================================
    # High-Level Intent API - How Chloe "thinks"
    # =========================================================================

    async def look_at_speaker(self):
        """Look at whoever is currently speaking.

        This is automatic when head tracking is active,
        but can be called explicitly.
        """
        self._attention = AttentionTarget.SPEAKER
        # Head tracker handles this automatically

    async def look_at_person(self, name: str):
        """Look at a specific remembered person."""
        if name in self._known_people:
            person = self._known_people[name]
            self._attention = AttentionTarget.PERSON

            # Convert their remembered DOA to pan angle
            pan = self.head_tracker.doa.doa_to_robot_pan(person.last_doa)
            await self._move_gantry(pan, 0, self.config.track_speed)

    async def look_direction(self, direction: str):
        """Look in a named direction.

        Args:
            direction: "left", "right", "up", "down", "forward", "behind"
        """
        directions = {
            "forward": (0, 0),
            "left": (-45, 0),
            "right": (45, 0),
            "up": (0, -30),
            "down": (0, 30),
            "behind": None,  # Special case - need base rotation
        }

        if direction == "behind":
            await self.turn_around()
            return

        pan, tilt = directions.get(direction, (0, 0))
        await self._move_gantry(pan, tilt, self.config.track_speed)

    async def turn_to_speaker(self):
        """Turn the entire robot to face the current speaker.

        Used when speaker is outside gantry range.
        """
        state = self.head_tracker.get_state()
        if state.location in (SpeakerLocation.NEED_BASE_TURN, SpeakerLocation.BEHIND):
            # Calculate rotation needed
            rotation = state.doa
            if rotation > 180:
                rotation -= 360

            await self._rotate_base(rotation)

            # After rotating, center the head
            await asyncio.sleep(0.5)
            await self._move_gantry(0, 0, self.config.track_speed)

    async def turn_around(self):
        """Turn 180 degrees (someone is behind)."""
        await self._rotate_base(180)
        await self._move_gantry(0, 0, self.config.track_speed)

    async def wave(self, arm: str = "right", style: str = "friendly"):
        """Wave hello or goodbye.

        Args:
            arm: "left" or "right"
            style: "friendly", "excited", "shy", "royal"
        """
        await self._gesture(Gesture.WAVE, arm=arm, style=style)

    async def point(self, direction: str, arm: str = "right"):
        """Point in a direction.

        Args:
            direction: "left", "right", "forward", "up", "down"
            arm: which arm to use
        """
        # Look where we're pointing
        await self.look_direction(direction)
        await self._gesture(Gesture.POINT, arm=arm, direction=direction)

    async def shrug(self):
        """Shrug shoulders (I don't know)."""
        await self._gesture(Gesture.SHRUG)

    async def express_thinking(self):
        """Express that Chloe is thinking."""
        # Look up slightly, maybe touch chin
        await self._move_gantry(self._current_pan, -15, 0.3)
        await self._gesture(Gesture.THINKING)

    async def express_excitement(self):
        """Express excitement (arms up, quick movement)."""
        await self._gesture(Gesture.EXCITED)

    async def go_home(self):
        """Return to neutral home position."""
        self._attention = AttentionTarget.REST
        await asyncio.gather(
            self._move_gantry(0, 0, self.config.track_speed),
            self._arms_to_pose("home")
        )

    async def center_head(self):
        """Just center the head (look forward)."""
        await self._move_gantry(0, 0, self.config.track_speed)

    # =========================================================================
    # Speech Integration - React while talking/listening
    # =========================================================================

    def set_speaking(self, is_speaking: bool):
        """Tell coordinator if Chloe is speaking.

        When speaking, we might want to:
        - Reduce head tracking responsiveness
        - Add subtle gestures
        """
        with self._lock:
            self._is_speaking = is_speaking

    async def add_speech_gesture(self, emotion: str = "neutral"):
        """Add subtle gesture during speech.

        Called by the TTS system to add natural movement.
        """
        gestures = {
            "happy": Gesture.EXCITED,
            "sad": Gesture.CALM,
            "curious": Gesture.THINKING,
            "greeting": Gesture.GREETING,
            "farewell": Gesture.FAREWELL,
        }
        gesture = gestures.get(emotion)
        if gesture:
            await self._gesture(gesture, subtle=True)

    # =========================================================================
    # Internal Handlers
    # =========================================================================

    def _on_speaker_detected(self, state: SpeakerState):
        """Handle speaker detection from head tracker."""
        with self._lock:
            self._current_speaker = PersonMemory(
                last_doa=state.doa,
                last_seen_time=time.time(),
                is_current_speaker=True
            )

    def _on_out_of_range(self, doa: float, location: SpeakerLocation):
        """Handle speaker outside gantry range."""
        # Log it - the high-level code should decide whether to turn
        print(f"Speaker out of range: {doa:.0f}° ({location.value})")

        # Optionally auto-turn (could be configurable)
        # asyncio.create_task(self.turn_to_speaker())

    # =========================================================================
    # Motor Control - Low Level
    # =========================================================================

    def _move_gantry_sync(self, pan: float, tilt: float, speed: float):
        """Synchronous gantry move (for head tracker callback)."""
        asyncio.create_task(self._move_gantry(pan, tilt, speed))

    async def _move_gantry(self, pan: float, tilt: float, speed: float):
        """Move the gantry to a position."""
        # Clamp to limits
        pan = max(self.config.pan_min, min(self.config.pan_max, pan))
        tilt = max(self.config.tilt_min, min(self.config.tilt_max, tilt))

        cmd = [
            "solo", "robo",
            "--port", self.config.right_port,
            "--ids", ",".join(map(str, self.config.gantry_ids)),
            "--positions", f"{pan},{tilt}",
            "--speed", str(speed)
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            self._current_pan = pan
            self._current_tilt = tilt
        except Exception as e:
            print(f"Gantry move error: {e}")

    async def _rotate_base(self, degrees: float):
        """Rotate the mecanum base.

        Args:
            degrees: Rotation angle (positive = clockwise/right)
        """
        # Mecanum wheels: rotate by driving opposite sides opposite directions
        # This is simplified - real implementation needs odometry

        # Estimate time based on rotation speed
        duration = abs(degrees) / 90.0  # ~1 sec per 90 degrees

        # Wheel velocities for rotation
        # [front, back_left, back_right]
        direction = 1 if degrees > 0 else -1
        velocities = [direction * 50, -direction * 50, direction * 50]

        cmd = [
            "solo", "robo",
            "--port", self.config.left_port,
            "--ids", ",".join(map(str, self.config.wheel_ids)),
            "--velocities", ",".join(map(str, velocities)),
            "--duration", str(duration)
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            self._base_angle = (self._base_angle + degrees) % 360
            print(f"Base rotated {degrees:.0f}° (estimated: {self._base_angle:.0f}°)")
        except Exception as e:
            print(f"Base rotation error: {e}")

    async def _arms_to_pose(self, pose: str, arm: str = "both"):
        """Move arms to a named pose."""
        poses = {
            "home": {
                "left": [0, -45, 90, 45, 0, 0],
                "right": [0, -45, 90, 45, 0, 0],
            },
            "wave": {
                "right": [0, -30, 120, 60, 0, 0],
            },
            "point_forward": {
                "right": [0, -60, 45, 90, 0, 0],
            },
            "arms_up": {
                "left": [0, -90, 45, 45, 0, 0],
                "right": [0, -90, 45, 45, 0, 0],
            },
            "arms_down": {
                "left": [0, 0, 90, 90, 0, 0],
                "right": [0, 0, 90, 90, 0, 0],
            },
        }

        if pose not in poses:
            return

        pose_data = poses[pose]
        tasks = []

        if arm in ("left", "both") and "left" in pose_data:
            tasks.append(self._move_arm("left", pose_data["left"]))

        if arm in ("right", "both") and "right" in pose_data:
            tasks.append(self._move_arm("right", pose_data["right"]))

        if tasks:
            await asyncio.gather(*tasks)

    async def _move_arm(self, arm: str, positions: List[float], speed: float = None):
        """Move an arm to specified joint positions."""
        speed = speed or self.config.gesture_speed

        port = self.config.left_port if arm == "left" else self.config.right_port
        ids = self.config.left_arm_ids if arm == "left" else self.config.right_arm_ids

        cmd = [
            "solo", "robo",
            "--port", port,
            "--ids", ",".join(map(str, ids)),
            "--positions", ",".join(map(str, positions)),
            "--speed", str(speed)
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
        except Exception as e:
            print(f"Arm move error: {e}")

    async def _gesture(self, gesture: Gesture, arm: str = "right",
                      style: str = "normal", direction: str = "forward",
                      subtle: bool = False):
        """Perform a gesture."""

        if gesture == Gesture.WAVE:
            await self._do_wave(arm, style, subtle)
        elif gesture == Gesture.POINT:
            await self._do_point(arm, direction)
        elif gesture == Gesture.SHRUG:
            await self._do_shrug(subtle)
        elif gesture == Gesture.THINKING:
            await self._do_thinking(subtle)
        elif gesture == Gesture.EXCITED:
            await self._do_excited(subtle)

    async def _do_wave(self, arm: str, style: str, subtle: bool):
        """Perform wave gesture."""
        # Move to wave position
        await self._arms_to_pose("wave", arm)

        # Wave motion
        wave_positions = [
            [0, -20, 130, 50, 30, 0],
            [0, -40, 110, 70, -30, 0],
        ]

        repetitions = 2 if not subtle else 1
        for _ in range(repetitions):
            for pos in wave_positions:
                await self._move_arm(arm, pos, speed=0.8)
                await asyncio.sleep(0.2 if not subtle else 0.3)

        # Return to home
        await self._arms_to_pose("home", arm)

    async def _do_point(self, arm: str, direction: str):
        """Point in a direction."""
        await self._arms_to_pose("point_forward", arm)
        await asyncio.sleep(1.0)
        await self._arms_to_pose("home", arm)

    async def _do_shrug(self, subtle: bool):
        """Shrug shoulders."""
        # Raise both arms slightly
        shrug_pos = [0, -60, 90, 60, 0, 0]
        await asyncio.gather(
            self._move_arm("left", shrug_pos, speed=0.8),
            self._move_arm("right", shrug_pos, speed=0.8)
        )
        await asyncio.sleep(0.5)
        await self._arms_to_pose("home")

    async def _do_thinking(self, subtle: bool):
        """Thinking gesture (hand to chin area)."""
        if subtle:
            # Just a slight head tilt
            return

        # Right hand toward face
        think_pos = [0, -80, 120, 90, 0, 0]
        await self._move_arm("right", think_pos, speed=0.5)
        await asyncio.sleep(1.5)
        await self._arms_to_pose("home", "right")

    async def _do_excited(self, subtle: bool):
        """Excited gesture (arms up)."""
        if subtle:
            # Small bounce
            up_pos = [0, -60, 80, 50, 0, 0]
        else:
            up_pos = [0, -100, 60, 40, 0, 0]

        await asyncio.gather(
            self._move_arm("left", up_pos, speed=0.9),
            self._move_arm("right", up_pos, speed=0.9)
        )
        await asyncio.sleep(0.5)
        await self._arms_to_pose("home")

    async def _disable_motors(self):
        """Disable all motor torques (emergency or shutdown)."""
        # Left bus
        cmd_left = [
            "solo", "robo",
            "--port", self.config.left_port,
            "--ids", ",".join(map(str,
                self.config.left_arm_ids +
                self.config.wheel_ids +
                (self.config.lift_id,)
            )),
            "--torque", "off"
        ]

        # Right bus
        cmd_right = [
            "solo", "robo",
            "--port", self.config.right_port,
            "--ids", ",".join(map(str,
                self.config.right_arm_ids +
                self.config.gantry_ids
            )),
            "--torque", "off"
        ]

        try:
            await asyncio.gather(
                asyncio.create_subprocess_exec(*cmd_left),
                asyncio.create_subprocess_exec(*cmd_right)
            )
        except Exception as e:
            print(f"Motor disable error: {e}")


# Singleton
_coordinator: Optional[MotionCoordinator] = None


def get_motion_coordinator() -> MotionCoordinator:
    """Get the singleton motion coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = MotionCoordinator()
    return _coordinator


# Test
if __name__ == "__main__":
    async def test():
        print("Testing Motion Coordinator")
        print("=" * 50)

        coord = get_motion_coordinator()
        await coord.start()

        print("\nSpeak to test head tracking...")
        print("Commands: wave, point, shrug, think, home, turn, quit")

        try:
            while True:
                cmd = input("\n> ").strip().lower()

                if cmd == "quit":
                    break
                elif cmd == "wave":
                    await coord.wave()
                elif cmd == "point":
                    await coord.point("forward")
                elif cmd == "shrug":
                    await coord.shrug()
                elif cmd == "think":
                    await coord.express_thinking()
                elif cmd == "home":
                    await coord.go_home()
                elif cmd == "turn":
                    await coord.turn_around()
                elif cmd.startswith("look "):
                    direction = cmd.split()[1]
                    await coord.look_direction(direction)
                else:
                    print("Unknown command")

        except KeyboardInterrupt:
            pass

        await coord.stop()
        print("\nTest complete!")

    asyncio.run(test())
