"""Base Robot Adapter Interface

This module defines the abstract interface for robot control adapters.
Implement this interface for each robot platform to enable the same
tools to work across different hardware.

The adapter pattern allows:
- Voice commands to work on any robot
- Easy migration between platforms
- Testing with mock adapters
- Fleet-wide command compatibility
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio


class Subsystem(Enum):
    """Robot subsystems that can be controlled independently."""
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    BASE = "base"
    LIFT = "lift"
    GANTRY = "gantry"
    GRIPPER_LEFT = "gripper_left"
    GRIPPER_RIGHT = "gripper_right"
    ALL = "all"  # For emergency stop


@dataclass
class ActionResult:
    """Result of a robot action execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0
    subsystem: Optional[Subsystem] = None

    @property
    def failed(self) -> bool:
        return not self.success


@dataclass
class ActionPrimitive:
    """A single atomic robot action.

    Actions can have dependencies (must wait for other actions)
    and can specify which subsystems they can run alongside.
    """
    name: str
    subsystem: str
    params: Dict[str, Any]
    timeout: float = 10.0
    dependencies: List[str] = field(default_factory=list)
    can_parallel: List[str] = field(default_factory=list)

    # Interrupt conditions - checked during execution
    interrupt_conditions: List[Callable[[], bool]] = field(default_factory=list)
    on_interrupt: Optional[Callable] = None


class RobotAdapter(ABC):
    """Abstract interface for robot control.

    Implement this for each robot platform to enable
    the same tools to work across different hardware.

    Example:
        adapter = ChloeAdapter()
        result = await adapter.execute(
            Subsystem.RIGHT_ARM,
            ActionPrimitive("wave", "right_arm", {"style": "friendly"})
        )
    """

    def __init__(self):
        self._stopped = False
        self._executing: Dict[Subsystem, asyncio.Task] = {}

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to robot hardware.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from robot hardware safely."""
        pass

    @abstractmethod
    async def execute(
        self, subsystem: Subsystem, action: ActionPrimitive
    ) -> ActionResult:
        """Execute an action on a subsystem.

        Args:
            subsystem: Which part of the robot to control
            action: The action to execute

        Returns:
            ActionResult with success status and details
        """
        pass

    @abstractmethod
    async def get_state(self, subsystem: Subsystem) -> Dict[str, Any]:
        """Get current state of a subsystem.

        Args:
            subsystem: Which subsystem to query

        Returns:
            Dictionary of state values (positions, velocities, etc.)
        """
        pass

    @abstractmethod
    async def stop(self, subsystem: Optional[Subsystem] = None) -> None:
        """Emergency stop.

        Args:
            subsystem: Specific subsystem to stop, or None for all
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return robot's capabilities for tool generation.

        Returns:
            Dictionary describing what the robot can do
        """
        pass

    @property
    def is_stopped(self) -> bool:
        """Check if robot is in emergency stop state."""
        return self._stopped

    async def reset_stop(self) -> bool:
        """Reset emergency stop state.

        Returns:
            True if reset successful
        """
        self._stopped = False
        return True

    async def move_to_pose(
        self, pose_name: str, subsystems: Optional[List[Subsystem]] = None
    ) -> ActionResult:
        """Move to a named pose.

        Args:
            pose_name: Name of the pose (e.g., "home", "wave")
            subsystems: Which subsystems to move, or None for all in pose

        Returns:
            ActionResult
        """
        # Default implementation - override for robot-specific poses
        return ActionResult(
            success=False,
            message=f"Pose '{pose_name}' not implemented for this adapter"
        )

    async def is_connected(self) -> bool:
        """Check if robot is connected.

        Returns:
            True if connected and ready
        """
        return False


class MockAdapter(RobotAdapter):
    """Mock adapter for testing without hardware.

    Records all actions for verification in tests.
    """

    def __init__(self):
        super().__init__()
        self.connected = False
        self.action_history: List[tuple] = []
        self.state: Dict[Subsystem, Dict] = {
            subsystem: {"position": [0] * 6, "velocity": [0] * 6}
            for subsystem in Subsystem
        }

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def disconnect(self) -> None:
        self.connected = False

    async def execute(
        self, subsystem: Subsystem, action: ActionPrimitive
    ) -> ActionResult:
        if self._stopped:
            return ActionResult(False, "Robot is stopped", subsystem=subsystem)

        self.action_history.append((subsystem, action))

        # Simulate execution time
        await asyncio.sleep(0.1)

        return ActionResult(
            success=True,
            message=f"Mock executed {action.name} on {subsystem.value}",
            subsystem=subsystem,
            duration_ms=100.0
        )

    async def get_state(self, subsystem: Subsystem) -> Dict[str, Any]:
        return self.state.get(subsystem, {})

    async def stop(self, subsystem: Optional[Subsystem] = None) -> None:
        self._stopped = True
        self.action_history.append((subsystem or Subsystem.ALL, "STOP"))

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "type": "mock",
            "arms": ["left", "right"],
            "gripper": True,
            "mobile_base": True,
            "lift": True,
            "gantry": True,
        }

    async def is_connected(self) -> bool:
        return self.connected
