"""Base Action Primitives

Defines the universal action interface that works across all robot platforms.
Each adapter implements these actions for its specific hardware.

Usage:
    from actions import UnifiedActions
    from robot_factory import create_robot

    robot = create_robot()
    actions = UnifiedActions(robot)

    await actions.wave()
    await actions.look_at(person)
    await actions.walk_to(x=1.0, y=0.0)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from enum import Enum
import asyncio

if TYPE_CHECKING:
    from robot_factory import Robot
    from cameras.base import SpatialDetection


class ActionStatus(Enum):
    """Status of an action execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Hand(Enum):
    """Which hand/arm to use."""
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class GestureType(Enum):
    """Pre-defined gesture types."""
    WAVE = "wave"
    POINT = "point"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    SHRUG = "shrug"
    NOD = "nod"
    SHAKE_HEAD = "shake_head"
    THINKING = "thinking"
    CELEBRATE = "celebrate"
    CLAP = "clap"
    BECKON = "beckon"
    STOP = "stop"
    OKAY = "okay"
    PEACE = "peace"
    FIST_BUMP = "fist_bump"


class EmotionType(Enum):
    """Emotional expressions through body language."""
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    CURIOUS = "curious"
    CONFUSED = "confused"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    ATTENTIVE = "attentive"


@dataclass
class ActionResult:
    """Result of an action execution."""
    status: ActionStatus
    action_name: str
    duration_ms: float = 0.0
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ActionStatus.COMPLETED


@dataclass
class Target:
    """A target for actions like look_at, point_at, walk_to."""
    # Angle-based (for look_at, point_at)
    angle: Optional[float] = None  # Degrees, 0 = forward, positive = right

    # Position-based (for walk_to, approach)
    x: Optional[float] = None  # Meters, forward
    y: Optional[float] = None  # Meters, left
    z: Optional[float] = None  # Meters, up

    # Person-based (from camera detection)
    detection: Optional['SpatialDetection'] = None

    # Named location
    location_name: Optional[str] = None

    @classmethod
    def from_angle(cls, angle: float) -> 'Target':
        return cls(angle=angle)

    @classmethod
    def from_position(cls, x: float, y: float, z: float = 0.0) -> 'Target':
        return cls(x=x, y=y, z=z)

    @classmethod
    def from_detection(cls, detection: 'SpatialDetection') -> 'Target':
        return cls(detection=detection)

    @classmethod
    def from_name(cls, name: str) -> 'Target':
        return cls(location_name=name)


class ActionCapability(Enum):
    """Capabilities a robot may or may not have."""
    WALK = "walk"
    WHEEL = "wheel"
    ARM_LEFT = "arm_left"
    ARM_RIGHT = "arm_right"
    GRIPPER_LEFT = "gripper_left"
    GRIPPER_RIGHT = "gripper_right"
    HEAD_PAN = "head_pan"
    HEAD_TILT = "head_tilt"
    TORSO = "torso"
    FINGERS = "fingers"


class ActionPrimitives(ABC):
    """Abstract base for action primitives.

    Each robot adapter should provide an implementation of this interface.
    Actions that the robot cannot perform should return ActionStatus.FAILED
    with an appropriate message.
    """

    @abstractmethod
    async def get_capabilities(self) -> List[ActionCapability]:
        """Return list of capabilities this robot has."""
        pass

    # === Gestures ===

    @abstractmethod
    async def wave(
        self,
        hand: Hand = Hand.RIGHT,
        duration: float = 2.0
    ) -> ActionResult:
        """Wave hello/goodbye."""
        pass

    @abstractmethod
    async def point_at(
        self,
        target: Target,
        hand: Hand = Hand.RIGHT,
        hold_duration: float = 1.0
    ) -> ActionResult:
        """Point at a target."""
        pass

    @abstractmethod
    async def gesture(
        self,
        gesture_type: GestureType,
        intensity: float = 1.0
    ) -> ActionResult:
        """Perform a named gesture."""
        pass

    @abstractmethod
    async def express(
        self,
        emotion: EmotionType,
        intensity: float = 1.0
    ) -> ActionResult:
        """Express an emotion through body language."""
        pass

    # === Head/Gaze ===

    @abstractmethod
    async def look_at(
        self,
        target: Target,
        speed: float = 0.5
    ) -> ActionResult:
        """Turn head to look at a target."""
        pass

    @abstractmethod
    async def look_forward(self) -> ActionResult:
        """Center the head to look straight ahead."""
        pass

    @abstractmethod
    async def nod(self, times: int = 1) -> ActionResult:
        """Nod the head (yes gesture)."""
        pass

    @abstractmethod
    async def shake_head(self, times: int = 1) -> ActionResult:
        """Shake the head (no gesture)."""
        pass

    # === Locomotion ===

    @abstractmethod
    async def walk_to(
        self,
        target: Target,
        speed: float = 0.5
    ) -> ActionResult:
        """Walk/drive to a target position."""
        pass

    @abstractmethod
    async def turn(
        self,
        angle: float,
        speed: float = 0.5
    ) -> ActionResult:
        """Turn in place by angle degrees."""
        pass

    @abstractmethod
    async def stop(self) -> ActionResult:
        """Emergency stop all movement."""
        pass

    @abstractmethod
    async def stand(self) -> ActionResult:
        """Stand up (bipedal) or ready position (wheeled)."""
        pass

    @abstractmethod
    async def sit(self) -> ActionResult:
        """Sit down (bipedal) or rest position."""
        pass

    # === Manipulation ===

    @abstractmethod
    async def reach(
        self,
        target: Target,
        hand: Hand = Hand.RIGHT
    ) -> ActionResult:
        """Reach toward a target."""
        pass

    @abstractmethod
    async def grab(
        self,
        hand: Hand = Hand.RIGHT
    ) -> ActionResult:
        """Close gripper to grab."""
        pass

    @abstractmethod
    async def release(
        self,
        hand: Hand = Hand.RIGHT
    ) -> ActionResult:
        """Open gripper to release."""
        pass

    @abstractmethod
    async def home_arms(self) -> ActionResult:
        """Return arms to home/rest position."""
        pass

    # === Compound Actions ===

    async def greet(
        self,
        target: Optional[Target] = None,
        style: str = "friendly"
    ) -> ActionResult:
        """Compound: look at person and wave."""
        results = []

        if target:
            results.append(await self.look_at(target))

        results.append(await self.wave())

        return ActionResult(
            status=ActionStatus.COMPLETED if all(r.success for r in results) else ActionStatus.FAILED,
            action_name="greet",
            message=f"Greeted with {len(results)} sub-actions"
        )

    async def acknowledge(self) -> ActionResult:
        """Compound: nod acknowledgment."""
        return await self.nod(times=1)

    async def decline(self) -> ActionResult:
        """Compound: shake head no."""
        return await self.shake_head(times=1)

    async def approach(
        self,
        target: Target,
        distance: float = 1.0
    ) -> ActionResult:
        """Compound: walk toward target, stopping at distance."""
        # Calculate position to stop at
        # This is a simple implementation; adapters can override
        return await self.walk_to(target, speed=0.3)
