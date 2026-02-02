"""Johnny Five Robot Adapter

This adapter controls a Johnny Five robot's hardware through Solo-CLI
and direct Dynamixel communication.

NOTE: "Johnny Five" is the robot MODEL. Each unit picks its own name.

Hardware Layout:
- ACM0: Left arm (1-6), lift (10), wheels (7-9)
- ACM1: Right arm (1-6), gantry (7-8)

All motors are Dynamixel XL330-M288-T running Protocol 2.0.
"""

import asyncio
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import struct

from .base import (
    RobotAdapter,
    Subsystem,
    ActionResult,
    ActionPrimitive,
)


@dataclass
class Johnny5HardwareConfig:
    """Hardware configuration for Johnny Five robots."""

    # Serial ports
    left_port: str = "/dev/ttyACM0"
    right_port: str = "/dev/ttyACM1"
    baudrate: int = 1_000_000

    # Motor IDs per subsystem
    left_arm_ids: tuple = (1, 2, 3, 4, 5, 6)
    right_arm_ids: tuple = (1, 2, 3, 4, 5, 6)
    wheel_ids: tuple = (7, 8, 9)
    lift_ids: tuple = (10,)
    gantry_ids: tuple = (7, 8)

    # Joint limits (degrees)
    arm_limits: tuple = (-150, 150)
    gantry_pan_limits: tuple = (-90, 90)
    gantry_tilt_limits: tuple = (-45, 45)
    lift_limits: tuple = (0, 300)  # mm


# Named poses for Johnny Five robots
JOHNNY5_POSES = {
    "home": {
        Subsystem.LEFT_ARM: [0, -45, 90, 45, 0, 0],
        Subsystem.RIGHT_ARM: [0, -45, 90, 45, 0, 0],
        Subsystem.GANTRY: [0, 0],
        Subsystem.LIFT: [150],
    },
    "wave": {
        Subsystem.RIGHT_ARM: [0, -30, 120, 60, 0, 0],
    },
    "wave_left": {
        Subsystem.LEFT_ARM: [0, -30, 120, 60, 0, 0],
    },
    "point": {
        Subsystem.RIGHT_ARM: [0, -60, 45, 90, 0, 0],
    },
    "ready_to_grab": {
        Subsystem.RIGHT_ARM: [0, -60, 90, 90, 0, 80],
    },
    "look_down": {
        Subsystem.GANTRY: [0, 30],
    },
    "look_up": {
        Subsystem.GANTRY: [0, -30],
    },
    "look_left": {
        Subsystem.GANTRY: [-45, 0],
    },
    "look_right": {
        Subsystem.GANTRY: [45, 0],
    },
    "arms_up": {
        Subsystem.LEFT_ARM: [0, -90, 45, 45, 0, 0],
        Subsystem.RIGHT_ARM: [0, -90, 45, 45, 0, 0],
    },
    "arms_down": {
        Subsystem.LEFT_ARM: [0, 0, 90, 90, 0, 0],
        Subsystem.RIGHT_ARM: [0, 0, 90, 90, 0, 0],
    },
}


class Johnny5Adapter(RobotAdapter):
    """Adapter for Johnny Five robots.

    Uses Solo-CLI for high-level commands and direct serial
    for low-latency emergency stop.
    """

    def __init__(self, config: Optional[Johnny5HardwareConfig] = None):
        super().__init__()
        self.config = config or Johnny5HardwareConfig()
        self._serial_left = None
        self._serial_right = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to the robot's motors via serial ports."""
        try:
            # For now, we'll use Solo-CLI which handles connection
            # Direct serial connection would be:
            # import serial
            # self._serial_left = serial.Serial(
            #     self.config.left_port,
            #     self.config.baudrate,
            #     timeout=0.1
            # )
            # self._serial_right = serial.Serial(
            #     self.config.right_port,
            #     self.config.baudrate,
            #     timeout=0.1
            # )

            # Test connection with Solo-CLI
            result = await self._run_solo_command(["solo", "robo", "--status"])
            self._connected = result.returncode == 0
            return self._connected

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Safely disconnect from the robot."""
        if self._connected:
            # Disable torque before disconnecting
            await self._disable_all_torque()
            self._connected = False

    async def execute(
        self, subsystem: Subsystem, action: ActionPrimitive
    ) -> ActionResult:
        """Execute an action on the robot.

        Routes actions to the appropriate handler based on action name.
        """
        if self._stopped:
            return ActionResult(
                success=False,
                message="Robot is in emergency stop state",
                subsystem=subsystem
            )

        start_time = asyncio.get_event_loop().time()

        try:
            # Route to specific handler
            if action.name == "move_to_position":
                result = await self._move_to_position(subsystem, action.params)
            elif action.name == "move_to_pose":
                result = await self._move_to_pose(subsystem, action.params)
            elif action.name == "gripper":
                result = await self._gripper_action(subsystem, action.params)
            elif action.name == "base_move":
                result = await self._base_move(action.params)
            elif action.name == "wave":
                result = await self._wave_action(subsystem, action.params)
            elif action.name == "look_at":
                result = await self._look_at(action.params)
            else:
                result = ActionResult(
                    success=False,
                    message=f"Unknown action: {action.name}"
                )

            result.duration_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000
            result.subsystem = subsystem
            return result

        except Exception as e:
            return ActionResult(
                success=False,
                message=str(e),
                subsystem=subsystem,
                duration_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )

    async def get_state(self, subsystem: Subsystem) -> Dict[str, Any]:
        """Get current state of a subsystem."""
        port, ids = self._get_port_and_ids(subsystem)

        # Would use direct Dynamixel read here
        # For now, return placeholder
        return {
            "positions": [0] * len(ids),
            "velocities": [0] * len(ids),
            "torques": [0] * len(ids),
            "connected": self._connected,
        }

    async def stop(self, subsystem: Optional[Subsystem] = None) -> None:
        """EMERGENCY STOP - Immediately disable torque.

        This bypasses all software logic and directly disables motors.
        """
        self._stopped = True
        print("!!! EMERGENCY STOP TRIGGERED !!!")

        # Direct torque disable - fastest possible
        if subsystem is None or subsystem == Subsystem.ALL:
            await self._disable_all_torque()
        else:
            port, ids = self._get_port_and_ids(subsystem)
            await self._disable_torque(port, ids)

    def get_capabilities(self) -> Dict[str, Any]:
        """Return Chloe's capabilities."""
        return {
            "type": "chloe",
            "name": "Johnny Five / Chloe",
            "arms": ["left", "right"],
            "arm_dof": 6,
            "gripper": True,
            "mobile_base": True,
            "base_type": "mecanum",
            "lift": True,
            "lift_range_mm": (0, 300),
            "gantry": True,
            "gantry_dof": 2,
            "camera": "OAK-D",
            "microphone": "ReSpeaker 4-Mic",
        }

    async def is_connected(self) -> bool:
        return self._connected

    async def move_to_pose(
        self, pose_name: str, subsystems: Optional[List[Subsystem]] = None
    ) -> ActionResult:
        """Move to a named pose."""
        if pose_name not in JOHNNY5_POSES:
            return ActionResult(
                success=False,
                message=f"Unknown pose: {pose_name}. Available: {list(JOHNNY5_POSES.keys())}"
            )

        pose = JOHNNY5_POSES[pose_name]
        results = []

        for subsystem, positions in pose.items():
            if subsystems is None or subsystem in subsystems:
                result = await self._move_to_position(
                    subsystem, {"positions": positions, "speed": 0.5}
                )
                results.append(result)

        success = all(r.success for r in results)
        return ActionResult(
            success=success,
            message=f"Moved to pose '{pose_name}'" if success else "Pose failed",
            data={"pose": pose_name, "results": results}
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_port_and_ids(self, subsystem: Subsystem) -> tuple:
        """Get serial port and motor IDs for a subsystem."""
        mapping = {
            Subsystem.LEFT_ARM: (self.config.left_port, self.config.left_arm_ids),
            Subsystem.RIGHT_ARM: (self.config.right_port, self.config.right_arm_ids),
            Subsystem.BASE: (self.config.left_port, self.config.wheel_ids),
            Subsystem.LIFT: (self.config.left_port, self.config.lift_ids),
            Subsystem.GANTRY: (self.config.right_port, self.config.gantry_ids),
            Subsystem.GRIPPER_LEFT: (self.config.left_port, (6,)),
            Subsystem.GRIPPER_RIGHT: (self.config.right_port, (6,)),
        }
        return mapping.get(subsystem, (self.config.left_port, ()))

    async def _run_solo_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a Solo-CLI command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return subprocess.CompletedProcess(
            cmd, process.returncode,
            stdout.decode() if stdout else "",
            stderr.decode() if stderr else ""
        )

    async def _move_to_position(
        self, subsystem: Subsystem, params: Dict
    ) -> ActionResult:
        """Move subsystem to specified positions."""
        positions = params.get("positions", [])
        speed = params.get("speed", 0.5)

        port, ids = self._get_port_and_ids(subsystem)

        if len(positions) != len(ids):
            return ActionResult(
                success=False,
                message=f"Position count ({len(positions)}) doesn't match motor count ({len(ids)})"
            )

        # Build Solo-CLI command
        cmd = [
            "solo", "robo",
            "--port", port,
            "--ids", ",".join(map(str, ids)),
            "--positions", ",".join(map(str, positions)),
            "--speed", str(speed)
        ]

        result = await self._run_solo_command(cmd)

        return ActionResult(
            success=result.returncode == 0,
            message=result.stdout or result.stderr or "Move complete"
        )

    async def _move_to_pose(
        self, subsystem: Subsystem, params: Dict
    ) -> ActionResult:
        """Move to a named pose."""
        pose_name = params.get("pose", "home")
        return await self.move_to_pose(pose_name, [subsystem])

    async def _gripper_action(
        self, subsystem: Subsystem, params: Dict
    ) -> ActionResult:
        """Open or close gripper."""
        action = params.get("action", "toggle")
        width = params.get("width", None)
        force = params.get("force", 0.5)

        # Gripper is motor 6 on each arm
        port, _ = self._get_port_and_ids(subsystem)
        gripper_id = 6

        if action == "open":
            position = 80  # Open position
        elif action == "close":
            position = 0   # Closed position
        elif action == "toggle":
            # Would need to read current position
            position = 80  # Default to open
        else:
            position = width if width is not None else 40

        cmd = [
            "solo", "robo",
            "--port", port,
            "--ids", str(gripper_id),
            "--positions", str(position),
            "--speed", "0.8"
        ]

        result = await self._run_solo_command(cmd)

        return ActionResult(
            success=result.returncode == 0,
            message=f"Gripper {action}" if result.returncode == 0 else result.stderr
        )

    async def _base_move(self, params: Dict) -> ActionResult:
        """Move the mecanum base."""
        direction = params.get("direction", "stop")
        speed = params.get("speed", 0.3)
        duration = params.get("duration", 1.0)

        # Mecanum wheel velocities for each direction
        # [front, back_left, back_right]
        velocities = {
            "forward": [1, 1, 1],
            "backward": [-1, -1, -1],
            "left": [-1, 1, -1],      # Strafe left
            "right": [1, -1, 1],       # Strafe right
            "rotate_left": [-1, 1, -1],
            "rotate_right": [1, -1, 1],
            "stop": [0, 0, 0],
        }

        vels = velocities.get(direction, [0, 0, 0])
        vels = [int(v * speed * 100) for v in vels]  # Scale by speed

        cmd = [
            "solo", "robo",
            "--port", self.config.left_port,
            "--ids", ",".join(map(str, self.config.wheel_ids)),
            "--velocities", ",".join(map(str, vels)),
            "--duration", str(duration)
        ]

        result = await self._run_solo_command(cmd)

        return ActionResult(
            success=result.returncode == 0,
            message=f"Base move {direction}"
        )

    async def _wave_action(
        self, subsystem: Subsystem, params: Dict
    ) -> ActionResult:
        """Perform a wave gesture."""
        style = params.get("style", "friendly")
        arm = params.get("arm", "right")

        # Determine which arm to use
        if arm == "right" or subsystem == Subsystem.RIGHT_ARM:
            target = Subsystem.RIGHT_ARM
            pose = "wave"
        else:
            target = Subsystem.LEFT_ARM
            pose = "wave_left"

        # Move to wave position
        result1 = await self.move_to_pose(pose, [target])
        if not result1.success:
            return result1

        # Do the wave motion (simplified)
        wave_positions = [
            [0, -20, 130, 50, 30, 0],
            [0, -40, 110, 70, -30, 0],
            [0, -20, 130, 50, 30, 0],
            [0, -40, 110, 70, -30, 0],
        ]

        for pos in wave_positions:
            await self._move_to_position(target, {"positions": pos, "speed": 0.8})
            await asyncio.sleep(0.3)

        # Return to home
        await self.move_to_pose("home", [target])

        return ActionResult(success=True, message=f"Waved {style}!")

    async def _look_at(self, params: Dict) -> ActionResult:
        """Point the gantry camera at a target."""
        target = params.get("target", "forward")
        pan = params.get("pan", None)
        tilt = params.get("tilt", None)

        # Named targets
        targets = {
            "forward": (0, 0),
            "left": (-45, 0),
            "right": (45, 0),
            "up": (0, -30),
            "down": (0, 30),
            "floor": (0, 45),
        }

        if pan is None or tilt is None:
            pan, tilt = targets.get(target, (0, 0))

        result = await self._move_to_position(
            Subsystem.GANTRY,
            {"positions": [pan, tilt], "speed": 0.5}
        )

        return ActionResult(
            success=result.success,
            message=f"Looking at {target}"
        )

    async def _disable_torque(self, port: str, ids: tuple) -> None:
        """Disable torque for specific motors."""
        cmd = [
            "solo", "robo",
            "--port", port,
            "--ids", ",".join(map(str, ids)),
            "--torque", "off"
        ]
        await self._run_solo_command(cmd)

    async def _disable_all_torque(self) -> None:
        """Disable torque on ALL motors."""
        # Disable both buses
        await asyncio.gather(
            self._disable_torque(
                self.config.left_port,
                self.config.left_arm_ids + self.config.wheel_ids + self.config.lift_ids
            ),
            self._disable_torque(
                self.config.right_port,
                self.config.right_arm_ids + self.config.gantry_ids
            )
        )
