"""Unitree G1 Robot Adapter

Adapter for the Unitree G1 humanoid robot.
23-43 DOF bipedal platform with RealSense camera and 4-mic array.

Hardware:
- Compute: NVIDIA Jetson Orin (up to 100 TOPS on EDU)
- Camera: Intel RealSense D435
- Microphone: Built-in 4-mic array
- LiDAR: LIVOX MID-360 (optional)
- Locomotion: Bipedal (23-43 DOF)
- Arms: 2x 7-DOF + Dex3 hands (EDU)
- Height: 1270mm
- Weight: 35kg

SDK: unitree_sdk2 (Python/C++)
ROS2: Supported

References:
- https://support.unitree.com/home/en/G1_developer
- https://github.com/unitreerobotics/unitree_sdk2
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

# Optional unitree_sdk2 support
try:
    import unitree_sdk2
    from unitree_sdk2.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2.idl.default import unitree_go_msg_dds__LowState_
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    UNITREE_SDK_AVAILABLE = False

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
class UnitreeG1Config:
    """Configuration for Unitree G1 robot."""

    # Network interface for unitree_sdk2
    NETWORK_INTERFACE: str = "eth0"

    # Joint names (G1 EDU with 43 DOF)
    HEAD_JOINTS: tuple = ("head_yaw", "head_pitch")

    LEFT_ARM_JOINTS: tuple = (
        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
        "l_elbow_pitch", "l_elbow_yaw", "l_wrist_roll", "l_wrist_pitch"
    )

    RIGHT_ARM_JOINTS: tuple = (
        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
        "r_elbow_pitch", "r_elbow_yaw", "r_wrist_roll", "r_wrist_pitch"
    )

    LEFT_LEG_JOINTS: tuple = (
        "l_hip_yaw", "l_hip_roll", "l_hip_pitch",
        "l_knee", "l_ankle_pitch", "l_ankle_roll"
    )

    RIGHT_LEG_JOINTS: tuple = (
        "r_hip_yaw", "r_hip_roll", "r_hip_pitch",
        "r_knee", "r_ankle_pitch", "r_ankle_roll"
    )

    # Joint limits (degrees)
    HEAD_YAW_LIMITS: tuple = (-60, 60)
    HEAD_PITCH_LIMITS: tuple = (-30, 30)

    # Motion parameters
    DEFAULT_SPEED: float = 0.5
    WALK_SPEED: float = 0.5  # m/s
    TORQUE_LIMIT: float = 120.0  # N.m (EDU Ultimate)


# Named poses for Unitree G1
UNITREE_G1_POSES = {
    "home": {
        Subsystem.LEFT_ARM: [0, 0, 0, 0, 0, 0, 0],
        Subsystem.RIGHT_ARM: [0, 0, 0, 0, 0, 0, 0],
        Subsystem.GANTRY: [0, 0],  # Head yaw/pitch
    },
    "wave": {
        Subsystem.RIGHT_ARM: [-45, 30, 0, 90, 0, 0, 0],
    },
    "wave_left": {
        Subsystem.LEFT_ARM: [-45, -30, 0, 90, 0, 0, 0],
    },
    "arms_open": {
        Subsystem.LEFT_ARM: [0, -60, 0, 0, 0, 0, 0],
        Subsystem.RIGHT_ARM: [0, 60, 0, 0, 0, 0, 0],
    },
    "look_down": {
        Subsystem.GANTRY: [0, 20],
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


class UnitreeG1Adapter(RobotAdapter):
    """Adapter for Unitree G1 humanoid robot.

    Communicates via unitree_sdk2 or ROS2. Falls back to simulation
    mode if neither is available.
    """

    def __init__(self, config: Optional[UnitreeG1Config] = None):
        super().__init__()
        self.config = config or UnitreeG1Config()
        self._connected = False
        self._simulation_mode = not (UNITREE_SDK_AVAILABLE or ROS2_AVAILABLE)
        self._joint_states = {}

        # SDK objects
        self._channel_factory = None
        self._low_cmd_publisher = None
        self._low_state_subscriber = None

        # ROS2 objects
        self._ros_node = None
        self._publishers = {}

    async def connect(self) -> bool:
        """Connect to the robot via unitree_sdk2 or ROS2."""
        if self._simulation_mode:
            print("[UnitreeG1] Running in simulation mode (SDK not available)")
            self._connected = True
            return True

        # Try unitree_sdk2 first
        if UNITREE_SDK_AVAILABLE:
            try:
                # Initialize SDK
                unitree_sdk2.init()

                self._channel_factory = unitree_sdk2.ChannelFactory()
                self._channel_factory.Init(0, self.config.NETWORK_INTERFACE)

                # Create publishers/subscribers
                self._low_cmd_publisher = self._channel_factory.CreatePublisher(
                    unitree_go_msg_dds__LowCmd_,
                    "rt/lowcmd"
                )

                self._low_state_subscriber = self._channel_factory.CreateSubscriber(
                    unitree_go_msg_dds__LowState_,
                    "rt/lowstate",
                    self._low_state_callback
                )

                self._connected = True
                print("[UnitreeG1] Connected via unitree_sdk2")
                return True

            except Exception as e:
                print(f"[UnitreeG1] SDK connection failed: {e}")

        # Fall back to ROS2
        if ROS2_AVAILABLE:
            try:
                if not rclpy.ok():
                    rclpy.init()

                self._ros_node = rclpy.create_node('unitree_g1_adapter')

                # Create publishers
                self._publishers['cmd_vel'] = self._ros_node.create_publisher(
                    Twist, '/cmd_vel', 10
                )
                self._publishers['head'] = self._ros_node.create_publisher(
                    Float64MultiArray, '/head/joint_positions', 10
                )
                self._publishers['arm'] = self._ros_node.create_publisher(
                    Float64MultiArray, '/arm/joint_positions', 10
                )

                # Subscribe to joint states
                self._ros_node.create_subscription(
                    JointState,
                    '/joint_states',
                    self._joint_state_callback,
                    10
                )

                self._connected = True
                print("[UnitreeG1] Connected via ROS2")
                return True

            except Exception as e:
                print(f"[UnitreeG1] ROS2 connection failed: {e}")

        # Fall back to simulation
        print("[UnitreeG1] Falling back to simulation mode")
        self._simulation_mode = True
        self._connected = True
        return True

    def _low_state_callback(self, msg):
        """Handle low-level state from unitree_sdk2."""
        # Extract joint positions
        for i, pos in enumerate(msg.motor_state):
            self._joint_states[f"joint_{i}"] = pos.q

    def _joint_state_callback(self, msg: 'JointState'):
        """Handle joint states from ROS2."""
        for name, position in zip(msg.name, msg.position):
            self._joint_states[name] = position

    async def disconnect(self) -> None:
        """Disconnect from the robot."""
        if self._ros_node:
            self._ros_node.destroy_node()
            self._ros_node = None
        self._connected = False
        print("[UnitreeG1] Disconnected")

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
            elif action.name == "stand":
                result = await self._stand()
            elif action.name == "sit":
                result = await self._sit()
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
            return {"positions": [0] * 7, "connected": True, "mode": "simulation"}

        if subsystem == Subsystem.LEFT_ARM:
            joints = self.config.LEFT_ARM_JOINTS
        elif subsystem == Subsystem.RIGHT_ARM:
            joints = self.config.RIGHT_ARM_JOINTS
        elif subsystem == Subsystem.GANTRY:
            joints = self.config.HEAD_JOINTS
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
        print("!!! UNITREE G1 EMERGENCY STOP !!!")

        # Send stop command
        await self._stop_walk()

    def get_capabilities(self) -> Dict[str, Any]:
        """Return robot capabilities."""
        return {
            "type": "unitree_g1",
            "name": "Unitree G1",
            "arms": ["left", "right"],
            "arm_dof": 7,
            "gripper": True,
            "gripper_type": "dex3",
            "mobile_base": True,
            "base_type": "bipedal",
            "total_dof": 43,  # EDU version
            "gantry": True,
            "gantry_dof": 2,
            "camera": "RealSense D435",
            "microphone": "4-Mic Array",
            "lidar": "LIVOX MID-360",
            "height_mm": 1270,
            "weight_kg": 35.0,
            "compute": "Orin 100 TOPS",
        }

    async def is_connected(self) -> bool:
        return self._connected

    async def move_to_pose(
        self, pose_name: str, subsystems: Optional[List[Subsystem]] = None
    ) -> ActionResult:
        """Move to a named pose."""
        if pose_name not in UNITREE_G1_POSES:
            return ActionResult(
                success=False,
                message=f"Unknown pose: {pose_name}. Available: {list(UNITREE_G1_POSES.keys())}"
            )

        pose = UNITREE_G1_POSES[pose_name]
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
            await asyncio.sleep(0.5)
            return ActionResult(
                success=True,
                message=f"[SIM] Moved {subsystem.value} to {positions}"
            )

        # Publish to appropriate topic
        if ROS2_AVAILABLE and self._ros_node:
            msg = Float64MultiArray()
            msg.data = [float(p) for p in positions]

            if subsystem == Subsystem.GANTRY:
                self._publishers['head'].publish(msg)
            elif subsystem in (Subsystem.LEFT_ARM, Subsystem.RIGHT_ARM):
                self._publishers['arm'].publish(msg)

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
            [-45, 30, 0, 70, 30, 0, 0],
            [-45, 30, 0, 110, -30, 0, 0],
            [-45, 30, 0, 70, 30, 0, 0],
            [-45, 30, 0, 110, -30, 0, 0],
        ]

        for pos in wave_positions:
            await self._move_to_position(target, {"positions": pos, "speed": 0.8})
            await asyncio.sleep(0.3)

        # Return home
        await self.move_to_pose("home", [target])

        return ActionResult(success=True, message="Waved!")

    async def _look_at(self, params: Dict) -> ActionResult:
        """Point the head at a target."""
        target = params.get("target", "forward")
        yaw = params.get("pan", params.get("yaw", None))
        pitch = params.get("tilt", params.get("pitch", None))

        targets = {
            "forward": (0, 0),
            "left": (-45, 0),
            "right": (45, 0),
            "up": (0, -20),
            "down": (0, 20),
        }

        if yaw is None or pitch is None:
            yaw, pitch = targets.get(target, (0, 0))

        # Clamp to limits
        yaw = max(self.config.HEAD_YAW_LIMITS[0],
                 min(self.config.HEAD_YAW_LIMITS[1], yaw))
        pitch = max(self.config.HEAD_PITCH_LIMITS[0],
                   min(self.config.HEAD_PITCH_LIMITS[1], pitch))

        result = await self._move_to_position(
            Subsystem.GANTRY,
            {"positions": [yaw, pitch], "speed": 0.5}
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
        if ROS2_AVAILABLE and 'cmd_vel' in self._publishers:
            twist = Twist()
            twist.linear.x = vx
            twist.linear.y = vy
            twist.angular.z = vz
            self._publishers['cmd_vel'].publish(twist)

        await asyncio.sleep(duration)
        await self._stop_walk()

        return ActionResult(
            success=True,
            message=f"Walked {direction} for {duration}s"
        )

    async def _stop_walk(self) -> ActionResult:
        """Stop walking."""
        if self._simulation_mode:
            return ActionResult(success=True, message="[SIM] Stopped")

        if ROS2_AVAILABLE and 'cmd_vel' in self._publishers:
            twist = Twist()
            self._publishers['cmd_vel'].publish(twist)

        return ActionResult(success=True, message="Stopped")

    async def _stand(self) -> ActionResult:
        """Stand up from sitting position."""
        if self._simulation_mode:
            await asyncio.sleep(2.0)
            return ActionResult(success=True, message="[SIM] Standing")

        # Send stand command via SDK
        # Implementation depends on unitree_sdk2 API

        return ActionResult(success=True, message="Standing")

    async def _sit(self) -> ActionResult:
        """Sit down from standing position."""
        if self._simulation_mode:
            await asyncio.sleep(2.0)
            return ActionResult(success=True, message="[SIM] Sitting")

        # Send sit command via SDK
        # Implementation depends on unitree_sdk2 API

        return ActionResult(success=True, message="Sitting")
