"""Booster K1 Robot Adapter

Adapter for the Booster K1 humanoid robot.
22 DOF bipedal platform with ZED camera and 6-mic array.

Hardware:
- Compute: NVIDIA Jetson Orin NX (40 TOPS)
- Camera: ZED X (GMSL2)
- Microphone: Circular 6-mic array
- Locomotion: Bipedal (22 DOF)
- Arms: 2x 6-DOF
- Height: 950mm
- Weight: 19.5kg

ROS2 Topics (typical Booster K1 setup):
- /joint_states: sensor_msgs/JointState
- /cmd_vel: geometry_msgs/Twist
- /head/pan_tilt: std_msgs/Float64MultiArray
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import (
    RobotAdapter,
    Subsystem,
    ActionResult,
    ActionPrimitive,
)

# Optional ROS2 support
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


@dataclass
class BoosterK1Config:
    """Configuration for Booster K1 robot."""

    # Joint names (typical Booster K1 configuration)
    HEAD_PAN: str = "head_pan"
    HEAD_TILT: str = "head_tilt"

    # Arm joint names (6 DOF each)
    LEFT_ARM_JOINTS: tuple = (
        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
        "l_elbow", "l_wrist_yaw", "l_wrist_roll"
    )
    RIGHT_ARM_JOINTS: tuple = (
        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
        "r_elbow", "r_wrist_yaw", "r_wrist_roll"
    )

    # Leg joint names (for walking)
    LEFT_LEG_JOINTS: tuple = (
        "l_hip_yaw", "l_hip_roll", "l_hip_pitch",
        "l_knee", "l_ankle_pitch", "l_ankle_roll"
    )
    RIGHT_LEG_JOINTS: tuple = (
        "r_hip_yaw", "r_hip_roll", "r_hip_pitch",
        "r_knee", "r_ankle_pitch", "r_ankle_roll"
    )

    # Joint limits (degrees)
    HEAD_PAN_LIMITS: tuple = (-90, 90)
    HEAD_TILT_LIMITS: tuple = (-30, 45)
    ARM_LIMITS: tuple = (-150, 150)

    # ROS2 topics
    CMD_VEL_TOPIC: str = "/cmd_vel"
    JOINT_STATE_TOPIC: str = "/joint_states"
    HEAD_CONTROL_TOPIC: str = "/head/pan_tilt"
    ARM_CONTROL_TOPIC: str = "/arm/joint_positions"

    # Motion parameters
    DEFAULT_SPEED: float = 0.5
    WALK_SPEED: float = 0.3  # m/s


# Named poses for Booster K1
BOOSTER_K1_POSES = {
    "home": {
        Subsystem.LEFT_ARM: [0, 0, 0, 0, 0, 0],
        Subsystem.RIGHT_ARM: [0, 0, 0, 0, 0, 0],
        Subsystem.GANTRY: [0, 0],  # Head pan/tilt
    },
    "wave": {
        Subsystem.RIGHT_ARM: [-30, 45, 0, 90, 0, 0],
    },
    "wave_left": {
        Subsystem.LEFT_ARM: [-30, -45, 0, 90, 0, 0],
    },
    "arms_open": {
        Subsystem.LEFT_ARM: [0, -60, 0, 0, 0, 0],
        Subsystem.RIGHT_ARM: [0, 60, 0, 0, 0, 0],
    },
    "look_down": {
        Subsystem.GANTRY: [0, 30],
    },
    "look_up": {
        Subsystem.GANTRY: [0, -20],
    },
    "look_left": {
        Subsystem.GANTRY: [-45, 0],
    },
    "look_right": {
        Subsystem.GANTRY: [45, 0],
    },
}


class BoosterK1Adapter(RobotAdapter):
    """Adapter for Booster K1 humanoid robot.

    Communicates via ROS2 topics. Falls back to simulation
    mode if ROS2 is not available.
    """

    def __init__(self, config: Optional[BoosterK1Config] = None):
        super().__init__()
        self.config = config or BoosterK1Config()
        self._connected = False
        self._ros_node = None
        self._publishers = {}
        self._joint_states = {}

        # Simulation mode if ROS2 not available
        self._simulation_mode = not ROS2_AVAILABLE

    async def connect(self) -> bool:
        """Connect to the robot via ROS2."""
        if self._simulation_mode:
            print("[BoosterK1] Running in simulation mode (ROS2 not available)")
            self._connected = True
            return True

        try:
            # Initialize ROS2
            if not rclpy.ok():
                rclpy.init()

            self._ros_node = rclpy.create_node('booster_k1_adapter')

            # Create publishers
            self._publishers['cmd_vel'] = self._ros_node.create_publisher(
                Twist, self.config.CMD_VEL_TOPIC, 10
            )
            self._publishers['head'] = self._ros_node.create_publisher(
                Float64MultiArray, self.config.HEAD_CONTROL_TOPIC, 10
            )
            self._publishers['arm'] = self._ros_node.create_publisher(
                Float64MultiArray, self.config.ARM_CONTROL_TOPIC, 10
            )

            # Subscribe to joint states
            self._ros_node.create_subscription(
                JointState,
                self.config.JOINT_STATE_TOPIC,
                self._joint_state_callback,
                10
            )

            self._connected = True
            print("[BoosterK1] Connected via ROS2")
            return True

        except Exception as e:
            print(f"[BoosterK1] Connection failed: {e}")
            print("[BoosterK1] Falling back to simulation mode")
            self._simulation_mode = True
            self._connected = True
            return True

    def _joint_state_callback(self, msg: 'JointState'):
        """Update joint states from ROS2."""
        for name, position in zip(msg.name, msg.position):
            self._joint_states[name] = position

    async def disconnect(self) -> None:
        """Disconnect from the robot."""
        if self._ros_node:
            self._ros_node.destroy_node()
            self._ros_node = None
        self._connected = False
        print("[BoosterK1] Disconnected")

    async def execute(
        self, subsystem: Subsystem, action: ActionPrimitive
    ) -> ActionResult:
        """Execute an action on the robot."""
        if self._stopped:
            return ActionResult(
                success=False,
                message="Robot is in emergency stop state",
                subsystem=subsystem
            )

        start_time = asyncio.get_event_loop().time()

        try:
            if action.name == "move_to_position":
                result = await self._move_to_position(subsystem, action.params)
            elif action.name == "move_to_pose":
                result = await self._move_to_pose(subsystem, action.params)
            elif action.name == "wave":
                result = await self._wave_action(subsystem, action.params)
            elif action.name == "look_at":
                result = await self._look_at(action.params)
            elif action.name == "walk":
                result = await self._walk(action.params)
            elif action.name == "stop_walk":
                result = await self._stop_walk()
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
        if self._simulation_mode:
            return {"positions": [0] * 6, "connected": True, "mode": "simulation"}

        if subsystem == Subsystem.LEFT_ARM:
            joints = self.config.LEFT_ARM_JOINTS
        elif subsystem == Subsystem.RIGHT_ARM:
            joints = self.config.RIGHT_ARM_JOINTS
        elif subsystem == Subsystem.GANTRY:
            joints = (self.config.HEAD_PAN, self.config.HEAD_TILT)
        else:
            return {"error": "Unknown subsystem"}

        positions = [self._joint_states.get(j, 0.0) for j in joints]
        return {
            "positions": positions,
            "joints": joints,
            "connected": self._connected,
        }

    async def stop(self, subsystem: Optional[Subsystem] = None) -> None:
        """Emergency stop."""
        self._stopped = True
        print("!!! BOOSTER K1 EMERGENCY STOP !!!")

        # Send zero velocity
        if not self._simulation_mode and 'cmd_vel' in self._publishers:
            twist = Twist()
            self._publishers['cmd_vel'].publish(twist)

    def get_capabilities(self) -> Dict[str, Any]:
        """Return robot capabilities."""
        return {
            "type": "booster_k1",
            "name": "Booster K1",
            "arms": ["left", "right"],
            "arm_dof": 6,
            "gripper": True,
            "mobile_base": True,
            "base_type": "bipedal",
            "total_dof": 22,
            "gantry": True,
            "gantry_dof": 2,
            "camera": "ZED X",
            "microphone": "Circular 6-Mic",
            "height_mm": 950,
            "weight_kg": 19.5,
            "compute": "Orin NX 40 TOPS",
        }

    async def is_connected(self) -> bool:
        return self._connected

    async def move_to_pose(
        self, pose_name: str, subsystems: Optional[List[Subsystem]] = None
    ) -> ActionResult:
        """Move to a named pose."""
        if pose_name not in BOOSTER_K1_POSES:
            return ActionResult(
                success=False,
                message=f"Unknown pose: {pose_name}. Available: {list(BOOSTER_K1_POSES.keys())}"
            )

        pose = BOOSTER_K1_POSES[pose_name]
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

    async def _move_to_position(
        self, subsystem: Subsystem, params: Dict
    ) -> ActionResult:
        """Move subsystem to specified positions."""
        positions = params.get("positions", [])
        speed = params.get("speed", 0.5)

        if self._simulation_mode:
            await asyncio.sleep(0.5)  # Simulate movement
            return ActionResult(
                success=True,
                message=f"[SIM] Moved {subsystem.value} to {positions}"
            )

        # Publish to appropriate topic
        msg = Float64MultiArray()
        msg.data = [float(p) for p in positions]

        if subsystem == Subsystem.GANTRY:
            self._publishers['head'].publish(msg)
        elif subsystem in (Subsystem.LEFT_ARM, Subsystem.RIGHT_ARM):
            self._publishers['arm'].publish(msg)

        # Wait for motion (simplified - real implementation would monitor joint states)
        await asyncio.sleep(1.0 / speed)

        return ActionResult(
            success=True,
            message=f"Moved {subsystem.value} to positions"
        )

    async def _move_to_pose(
        self, subsystem: Subsystem, params: Dict
    ) -> ActionResult:
        """Move to a named pose."""
        pose_name = params.get("pose", "home")
        return await self.move_to_pose(pose_name, [subsystem])

    async def _wave_action(
        self, subsystem: Subsystem, params: Dict
    ) -> ActionResult:
        """Perform a wave gesture."""
        arm = params.get("arm", "right")
        style = params.get("style", "friendly")

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

        # Wave motion
        wave_positions = [
            [-30, 45, 0, 70, 30, 0],
            [-30, 45, 0, 110, -30, 0],
            [-30, 45, 0, 70, 30, 0],
            [-30, 45, 0, 110, -30, 0],
        ]

        for pos in wave_positions:
            await self._move_to_position(target, {"positions": pos, "speed": 0.8})
            await asyncio.sleep(0.3)

        # Return home
        await self.move_to_pose("home", [target])

        return ActionResult(success=True, message=f"Waved {style}!")

    async def _look_at(self, params: Dict) -> ActionResult:
        """Point the head at a target."""
        target = params.get("target", "forward")
        pan = params.get("pan", None)
        tilt = params.get("tilt", None)

        targets = {
            "forward": (0, 0),
            "left": (-45, 0),
            "right": (45, 0),
            "up": (0, -20),
            "down": (0, 30),
        }

        if pan is None or tilt is None:
            pan, tilt = targets.get(target, (0, 0))

        # Clamp to limits
        pan = max(self.config.HEAD_PAN_LIMITS[0],
                 min(self.config.HEAD_PAN_LIMITS[1], pan))
        tilt = max(self.config.HEAD_TILT_LIMITS[0],
                  min(self.config.HEAD_TILT_LIMITS[1], tilt))

        result = await self._move_to_position(
            Subsystem.GANTRY,
            {"positions": [pan, tilt], "speed": 0.5}
        )

        return ActionResult(
            success=result.success,
            message=f"Looking at {target}"
        )

    async def _walk(self, params: Dict) -> ActionResult:
        """Command the robot to walk."""
        direction = params.get("direction", "forward")
        speed = params.get("speed", self.config.WALK_SPEED)
        duration = params.get("duration", 2.0)

        velocities = {
            "forward": (speed, 0, 0),
            "backward": (-speed, 0, 0),
            "left": (0, speed, 0),
            "right": (0, -speed, 0),
            "rotate_left": (0, 0, 0.5),
            "rotate_right": (0, 0, -0.5),
        }

        vx, vy, vz = velocities.get(direction, (0, 0, 0))

        if self._simulation_mode:
            await asyncio.sleep(duration)
            return ActionResult(
                success=True,
                message=f"[SIM] Walked {direction} for {duration}s"
            )

        # Publish velocity
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = vz

        self._publishers['cmd_vel'].publish(twist)

        await asyncio.sleep(duration)

        # Stop
        await self._stop_walk()

        return ActionResult(
            success=True,
            message=f"Walked {direction} for {duration}s"
        )

    async def _stop_walk(self) -> ActionResult:
        """Stop walking."""
        if self._simulation_mode:
            return ActionResult(success=True, message="[SIM] Stopped")

        twist = Twist()
        self._publishers['cmd_vel'].publish(twist)

        return ActionResult(success=True, message="Stopped")
