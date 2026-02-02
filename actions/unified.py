"""Unified Actions Interface

Provides a single interface for actions across all robot platforms.
Automatically routes actions to the appropriate adapter implementation.

Usage:
    from robot_factory import create_robot
    from actions import UnifiedActions

    robot = create_robot()
    await robot.adapter.connect()

    actions = UnifiedActions(robot)

    # These work on any robot body
    await actions.wave()
    await actions.look_at(Target.from_angle(45))
    await actions.gesture(GestureType.THUMBS_UP)
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
import asyncio
import time

from .base import (
    ActionPrimitives,
    ActionResult,
    ActionStatus,
    ActionCapability,
    Target,
    Hand,
    GestureType,
    EmotionType,
)

if TYPE_CHECKING:
    from robot_factory import Robot


class UnifiedActions(ActionPrimitives):
    """Unified action interface that works across all robot bodies.

    This class delegates actions to the robot's adapter, providing
    a consistent API regardless of the underlying hardware.

    Features:
    - Automatic capability detection
    - Graceful degradation for unsupported actions
    - Action queuing and coordination
    - Logging and metrics
    """

    def __init__(self, robot: 'Robot'):
        """Initialize with a robot instance.

        Args:
            robot: Robot from robot_factory.create_robot()
        """
        self.robot = robot
        self.adapter = robot.adapter
        self._capabilities: Optional[List[ActionCapability]] = None
        self._action_count = 0
        self._last_action_time = 0.0

    async def get_capabilities(self) -> List[ActionCapability]:
        """Get capabilities from the robot adapter."""
        if self._capabilities is None:
            self._capabilities = await self._detect_capabilities()
        return self._capabilities

    async def _detect_capabilities(self) -> List[ActionCapability]:
        """Detect capabilities based on robot config."""
        caps = []
        config = self.robot.config

        # Locomotion
        if config.mobile_base:
            if config.base_type == "bipedal":
                caps.append(ActionCapability.WALK)
            else:
                caps.append(ActionCapability.WHEEL)

        # Arms
        if config.arms >= 1:
            caps.append(ActionCapability.ARM_RIGHT)
            if config.grippers:
                caps.append(ActionCapability.GRIPPER_RIGHT)
        if config.arms >= 2:
            caps.append(ActionCapability.ARM_LEFT)
            if config.grippers:
                caps.append(ActionCapability.GRIPPER_LEFT)

        # Head
        caps.append(ActionCapability.HEAD_PAN)
        caps.append(ActionCapability.HEAD_TILT)

        return caps

    def _log_action(self, name: str, result: ActionResult):
        """Log action execution."""
        self._action_count += 1
        self._last_action_time = time.time()
        status = "OK" if result.success else "FAIL"
        print(f"[Actions] {name}: {status} ({result.duration_ms:.0f}ms)")

    async def _execute_via_adapter(
        self,
        action_name: str,
        subsystem: str,
        params: Dict[str, Any]
    ) -> ActionResult:
        """Execute action through the adapter layer."""
        from adapters.base import Subsystem, ActionPrimitive

        start = time.time()

        try:
            # Map subsystem string to enum
            subsystem_map = {
                "gantry": Subsystem.GANTRY,
                "head": Subsystem.GANTRY,  # Alias
                "left_arm": Subsystem.LEFT_ARM,
                "right_arm": Subsystem.RIGHT_ARM,
                "base": Subsystem.BASE,
                "lift": Subsystem.LIFT,
            }

            sub = subsystem_map.get(subsystem, Subsystem.GANTRY)

            action = ActionPrimitive(
                name=action_name,
                subsystem=subsystem,
                params=params,
                timeout=5.0
            )

            adapter_result = await self.adapter.execute(sub, action)

            duration = (time.time() - start) * 1000

            result = ActionResult(
                status=ActionStatus.COMPLETED if adapter_result.success else ActionStatus.FAILED,
                action_name=action_name,
                duration_ms=duration,
                message=adapter_result.message,
                data=adapter_result.data
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            result = ActionResult(
                status=ActionStatus.FAILED,
                action_name=action_name,
                duration_ms=duration,
                message=str(e)
            )

        self._log_action(action_name, result)
        return result

    # === Gestures ===

    async def wave(
        self,
        hand: Hand = Hand.RIGHT,
        duration: float = 2.0
    ) -> ActionResult:
        """Wave hello/goodbye."""
        return await self._execute_via_adapter(
            "wave",
            "right_arm" if hand != Hand.LEFT else "left_arm",
            {"hand": hand.value, "duration": duration}
        )

    async def point_at(
        self,
        target: Target,
        hand: Hand = Hand.RIGHT,
        hold_duration: float = 1.0
    ) -> ActionResult:
        """Point at a target."""
        params = {
            "hand": hand.value,
            "hold_duration": hold_duration
        }

        if target.angle is not None:
            params["angle"] = target.angle
        if target.detection is not None:
            params["angle"] = target.detection.angle
            params["distance"] = target.detection.distance

        return await self._execute_via_adapter(
            "point_at",
            "right_arm" if hand != Hand.LEFT else "left_arm",
            params
        )

    async def gesture(
        self,
        gesture_type: GestureType,
        intensity: float = 1.0
    ) -> ActionResult:
        """Perform a named gesture."""
        # Map gestures to specific implementations
        if gesture_type == GestureType.WAVE:
            return await self.wave()
        elif gesture_type == GestureType.NOD:
            return await self.nod()
        elif gesture_type == GestureType.SHAKE_HEAD:
            return await self.shake_head()
        elif gesture_type == GestureType.THUMBS_UP:
            return await self._execute_via_adapter(
                "thumbs_up", "right_arm", {"intensity": intensity}
            )
        elif gesture_type == GestureType.SHRUG:
            return await self._execute_via_adapter(
                "shrug", "right_arm", {"intensity": intensity}
            )
        else:
            return await self._execute_via_adapter(
                "gesture", "right_arm",
                {"type": gesture_type.value, "intensity": intensity}
            )

    async def express(
        self,
        emotion: EmotionType,
        intensity: float = 1.0
    ) -> ActionResult:
        """Express an emotion through body language."""
        return await self._execute_via_adapter(
            "express", "right_arm",
            {"emotion": emotion.value, "intensity": intensity}
        )

    # === Head/Gaze ===

    async def look_at(
        self,
        target: Target,
        speed: float = 0.5
    ) -> ActionResult:
        """Turn head to look at a target."""
        params = {"speed": speed}

        if target.angle is not None:
            params["pan"] = target.angle
            params["tilt"] = 0.0
        elif target.detection is not None:
            params["pan"] = target.detection.angle
            # Calculate tilt based on y position if available
            params["tilt"] = 0.0

        return await self._execute_via_adapter("look_at", "gantry", params)

    async def look_forward(self) -> ActionResult:
        """Center the head to look straight ahead."""
        return await self._execute_via_adapter(
            "look_at", "gantry", {"pan": 0.0, "tilt": 0.0}
        )

    async def nod(self, times: int = 1) -> ActionResult:
        """Nod the head (yes gesture)."""
        return await self._execute_via_adapter(
            "nod", "gantry", {"times": times}
        )

    async def shake_head(self, times: int = 1) -> ActionResult:
        """Shake the head (no gesture)."""
        return await self._execute_via_adapter(
            "shake_head", "gantry", {"times": times}
        )

    # === Locomotion ===

    async def walk_to(
        self,
        target: Target,
        speed: float = 0.5
    ) -> ActionResult:
        """Walk/drive to a target position."""
        params = {"speed": speed}

        if target.x is not None:
            params["x"] = target.x
            params["y"] = target.y or 0.0
        elif target.detection is not None:
            # Convert detection to position
            params["x"] = target.detection.z / 1000.0  # mm to m
            params["y"] = -target.detection.x / 1000.0

        return await self._execute_via_adapter("walk_to", "base", params)

    async def turn(
        self,
        angle: float,
        speed: float = 0.5
    ) -> ActionResult:
        """Turn in place by angle degrees."""
        return await self._execute_via_adapter(
            "turn", "base", {"angle": angle, "speed": speed}
        )

    async def stop(self) -> ActionResult:
        """Emergency stop all movement."""
        return await self._execute_via_adapter("stop", "base", {})

    async def stand(self) -> ActionResult:
        """Stand up (bipedal) or ready position (wheeled)."""
        return await self._execute_via_adapter("stand", "base", {})

    async def sit(self) -> ActionResult:
        """Sit down (bipedal) or rest position."""
        return await self._execute_via_adapter("sit", "base", {})

    # === Manipulation ===

    async def reach(
        self,
        target: Target,
        hand: Hand = Hand.RIGHT
    ) -> ActionResult:
        """Reach toward a target."""
        params = {"hand": hand.value}

        if target.x is not None:
            params["x"] = target.x
            params["y"] = target.y or 0.0
            params["z"] = target.z or 0.0

        return await self._execute_via_adapter(
            "reach",
            "right_arm" if hand != Hand.LEFT else "left_arm",
            params
        )

    async def grab(self, hand: Hand = Hand.RIGHT) -> ActionResult:
        """Close gripper to grab."""
        return await self._execute_via_adapter(
            "grab",
            "right_arm" if hand != Hand.LEFT else "left_arm",
            {"hand": hand.value}
        )

    async def release(self, hand: Hand = Hand.RIGHT) -> ActionResult:
        """Open gripper to release."""
        return await self._execute_via_adapter(
            "release",
            "right_arm" if hand != Hand.LEFT else "left_arm",
            {"hand": hand.value}
        )

    async def home_arms(self) -> ActionResult:
        """Return arms to home/rest position."""
        results = []
        results.append(await self._execute_via_adapter("home", "right_arm", {}))
        results.append(await self._execute_via_adapter("home", "left_arm", {}))

        return ActionResult(
            status=ActionStatus.COMPLETED if all(r.success for r in results) else ActionStatus.FAILED,
            action_name="home_arms",
            message="Arms returned to home position"
        )

    # === Convenience Methods ===

    def can(self, capability: ActionCapability) -> bool:
        """Check if robot has a capability (sync version)."""
        # Use cached capabilities if available
        if self._capabilities is None:
            # Detect synchronously from config
            config = self.robot.config
            if capability == ActionCapability.WALK:
                return config.base_type == "bipedal"
            if capability == ActionCapability.WHEEL:
                return config.base_type in ("wheeled", "tracks")
            if capability in (ActionCapability.ARM_LEFT, ActionCapability.ARM_RIGHT):
                return config.arms >= 1
            return True
        return capability in self._capabilities

    @property
    def robot_name(self) -> str:
        """Get the robot's name."""
        return self.robot.name

    @property
    def action_count(self) -> int:
        """Total actions executed."""
        return self._action_count
