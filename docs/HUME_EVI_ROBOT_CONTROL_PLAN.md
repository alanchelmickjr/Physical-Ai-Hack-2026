# Hume EVI + Robot Control Integration Plan

## Executive Summary

Give Chloe (Johnny Five) the ability to control her own body through voice commands by integrating Hume EVI's tool calling system with a robust robot control abstraction layer. The system must handle sequential and parallel tool execution with dependency management.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER VOICE INPUT                             │
│                     "Chloe, pick up the red block"                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         HUME EVI (Cloud)                             │
│  • Speech-to-text                                                    │
│  • Intent understanding                                              │
│  • Tool selection & argument inference                               │
│  • Response generation                                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                          tool_call message
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TOOL EXECUTION ENGINE (Local)                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │               DEPENDENCY RESOLVER & SCHEDULER                │    │
│  │  • Parse tool call into action graph                        │    │
│  │  • Identify parallel vs sequential requirements             │    │
│  │  • Execute with proper ordering                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │
│  │  ROBOT ADAPTER  │ │  ROBOT ADAPTER  │ │  ROBOT ADAPTER  │       │
│  │   (Chloe/J5)    │ │   (Future Bot)  │ │    (Generic)    │       │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     HARDWARE ABSTRACTION LAYER                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Solo-CLI   │ │  LeRobot    │ │  Direct     │ │   Custom    │   │
│  │  (SO-101)   │ │  (ACT/VLA)  │ │  Serial     │ │   Driver    │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         PHYSICAL HARDWARE                            │
│  ACM0: Left arm (1-6), lift (10), wheels (7-9)                      │
│  ACM1: Right arm (1-6), gantry (7-8)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Forking Strategy

### 1.1 Fork Solo-CLI

**Repository:** https://github.com/TheRobotStudio/SO-ARM100 (or solo-cli source)

**Why Fork:**
- Add voice-triggered command interface
- Customize for our motor IDs and bus layout
- Add safety limits for Chloe's specific hardware
- Integrate with our tool execution engine

**Fork Name:** `chloe-solo-cli` or `johnny5-solo`

**Key Modifications:**
```python
# Current solo-cli usage
solo robo --motors all
solo robo --teleop

# Our additions
solo robo --voice-mode          # Listen for tool commands
solo robo --json-api            # Accept JSON action commands
solo robo --safety-limits       # Enforce Chloe's joint limits
```

### 1.2 Fork LeRobot AlohaMini

**Repository:** https://github.com/liyiteng/lerobot_alohamini

**Why Fork:**
- Different motor IDs than default AlohaMini
- Custom wheel arrangement (mecanum)
- Different lift motor configuration
- Add Hume EVI integration layer

**Fork Name:** `chloe-lerobot` or `johnny5-lerobot`

**Key Modifications:**
```yaml
# configs/robot/chloe.yaml
robot:
  name: chloe

  arms:
    left:
      port: /dev/ttyACM0
      motor_ids: [1, 2, 3, 4, 5, 6]
    right:
      port: /dev/ttyACM1
      motor_ids: [1, 2, 3, 4, 5, 6]

  base:
    port: /dev/ttyACM0
    motor_ids: [7, 8, 9]  # Mecanum wheels
    type: mecanum

  lift:
    port: /dev/ttyACM0
    motor_id: 10

  gantry:
    port: /dev/ttyACM1
    motor_ids: [7, 8]  # Pan, tilt
```

---

## Part 2: Tool Execution Engine

### 2.1 Core Concepts

**Action Primitives** - Atomic robot operations:
```python
@dataclass
class ActionPrimitive:
    name: str                    # e.g., "move_arm"
    subsystem: str               # e.g., "left_arm", "right_arm", "base"
    params: dict                 # e.g., {"position": [x,y,z], "speed": 0.5}
    timeout: float               # Max execution time
    dependencies: list[str]      # Actions that must complete first
    can_parallel: list[str]      # Subsystems this can run alongside
```

**Execution Modes:**
| Mode | Description | Example |
|------|-------------|---------|
| Sequential | One after another | Move arm → Close gripper |
| Parallel | Simultaneous | Both arms wave |
| Dependent | Wait for condition | Close gripper AFTER arm at position |
| Interruptible | Can be stopped mid-action | Emergency stop |

### 2.2 Dependency Graph

```python
class ActionGraph:
    """Directed acyclic graph of robot actions"""

    def __init__(self):
        self.actions: dict[str, ActionPrimitive] = {}
        self.dependencies: dict[str, set[str]] = {}  # action -> depends_on

    def add_action(self, action: ActionPrimitive):
        self.actions[action.name] = action
        self.dependencies[action.name] = set(action.dependencies)

    def get_execution_order(self) -> list[list[str]]:
        """Returns actions grouped by execution wave.

        Actions in the same wave can execute in parallel.
        Each wave must complete before the next begins.
        """
        # Topological sort with parallel grouping
        remaining = set(self.actions.keys())
        waves = []

        while remaining:
            # Find actions with all dependencies satisfied
            ready = {
                action for action in remaining
                if self.dependencies[action].issubset(
                    set(self.actions.keys()) - remaining
                )
            }

            # Group by subsystem for parallel execution
            wave = self._group_parallelizable(ready)
            waves.append(wave)
            remaining -= ready

        return waves
```

### 2.3 Execution Engine

```python
class ToolExecutionEngine:
    """Executes robot tools with dependency management"""

    def __init__(self, robot_adapter: RobotAdapter):
        self.robot = robot_adapter
        self.executing: dict[str, asyncio.Task] = {}
        self.completed: set[str] = set()
        self.lock = asyncio.Lock()

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call from Hume EVI"""

        # Parse tool into action graph
        graph = self.parse_tool_to_graph(tool_call)

        # Execute waves
        results = []
        for wave in graph.get_execution_order():
            wave_results = await self._execute_wave(wave)
            results.extend(wave_results)

            # Check for failures
            if any(r.failed for r in wave_results):
                return ToolResult(
                    success=False,
                    message=f"Action failed: {wave_results}",
                    partial_results=results
                )

        return ToolResult(success=True, results=results)

    async def _execute_wave(self, actions: list[ActionPrimitive]) -> list:
        """Execute a wave of parallel-safe actions"""

        # Group by subsystem
        by_subsystem = defaultdict(list)
        for action in actions:
            by_subsystem[action.subsystem].append(action)

        # Execute each subsystem's actions (parallel across subsystems)
        tasks = []
        for subsystem, subsystem_actions in by_subsystem.items():
            task = asyncio.create_task(
                self._execute_subsystem_actions(subsystem, subsystem_actions)
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def _execute_subsystem_actions(
        self, subsystem: str, actions: list[ActionPrimitive]
    ):
        """Execute actions on a single subsystem (sequential within subsystem)"""
        results = []
        for action in actions:
            result = await self.robot.execute(subsystem, action)
            results.append(result)
            if result.failed:
                break
        return results
```

### 2.4 Example: "Pick up the red block"

```python
# Tool call from Hume EVI
tool_call = {
    "name": "pick_object",
    "arguments": {
        "object": "red block",
        "hand": "right"
    }
}

# Parsed into action graph:
actions = [
    ActionPrimitive(
        name="look_at_object",
        subsystem="gantry",
        params={"target": "red block"},
        dependencies=[],
        can_parallel=["left_arm", "right_arm"]
    ),
    ActionPrimitive(
        name="move_arm_to_object",
        subsystem="right_arm",
        params={"target": "red block", "offset": [0, 0, 0.1]},
        dependencies=["look_at_object"],  # Wait for gantry
        can_parallel=["left_arm", "gantry"]
    ),
    ActionPrimitive(
        name="open_gripper",
        subsystem="right_arm",
        params={"width": 0.08},
        dependencies=["move_arm_to_object"],
        can_parallel=["left_arm"]
    ),
    ActionPrimitive(
        name="lower_to_object",
        subsystem="right_arm",
        params={"target": "red block", "offset": [0, 0, 0]},
        dependencies=["open_gripper"],
        can_parallel=["left_arm"]
    ),
    ActionPrimitive(
        name="close_gripper",
        subsystem="right_arm",
        params={"force": 0.5},
        dependencies=["lower_to_object"],
        can_parallel=["left_arm"]
    ),
    ActionPrimitive(
        name="lift_object",
        subsystem="right_arm",
        params={"height": 0.15},
        dependencies=["close_gripper"],
        can_parallel=["left_arm", "gantry"]
    ),
]

# Execution waves:
# Wave 1: [look_at_object]                    - Gantry moves
# Wave 2: [move_arm_to_object]                - Arm moves (after gantry)
# Wave 3: [open_gripper]                      - Sequential on arm
# Wave 4: [lower_to_object]                   - Sequential on arm
# Wave 5: [close_gripper]                     - Sequential on arm
# Wave 6: [lift_object]                       - Sequential on arm
```

---

## Part 3: Hume EVI Tool Integration

### 3.1 Tool Definition Schema

Define tools in Hume EVI configuration:

```json
{
  "tools": [
    {
      "name": "move_arm",
      "description": "Move robot arm to a position or named pose",
      "parameters": {
        "type": "object",
        "properties": {
          "arm": {
            "type": "string",
            "enum": ["left", "right", "both"],
            "description": "Which arm to move"
          },
          "position": {
            "type": "string",
            "description": "Named pose (home, wave, point, grab) or XYZ coordinates"
          },
          "speed": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 1.0,
            "default": 0.5,
            "description": "Movement speed (0.1=slow, 1.0=fast)"
          }
        },
        "required": ["arm", "position"]
      }
    },
    {
      "name": "pick_object",
      "description": "Pick up an object that Chloe can see",
      "parameters": {
        "type": "object",
        "properties": {
          "object": {
            "type": "string",
            "description": "Description of the object to pick up"
          },
          "hand": {
            "type": "string",
            "enum": ["left", "right"],
            "default": "right"
          }
        },
        "required": ["object"]
      }
    },
    {
      "name": "wave",
      "description": "Wave hello or goodbye",
      "parameters": {
        "type": "object",
        "properties": {
          "arm": {
            "type": "string",
            "enum": ["left", "right", "both"],
            "default": "right"
          },
          "style": {
            "type": "string",
            "enum": ["friendly", "excited", "royal"],
            "default": "friendly"
          }
        }
      }
    },
    {
      "name": "look_at",
      "description": "Turn head/gantry to look at something",
      "parameters": {
        "type": "object",
        "properties": {
          "target": {
            "type": "string",
            "description": "What to look at: person name, object, or direction"
          }
        },
        "required": ["target"]
      }
    },
    {
      "name": "move_base",
      "description": "Move the robot base",
      "parameters": {
        "type": "object",
        "properties": {
          "direction": {
            "type": "string",
            "enum": ["forward", "backward", "left", "right", "rotate_left", "rotate_right"]
          },
          "distance": {
            "type": "number",
            "description": "Distance in meters (or degrees for rotation)"
          }
        },
        "required": ["direction"]
      }
    },
    {
      "name": "stop",
      "description": "Emergency stop all movement",
      "parameters": {
        "type": "object",
        "properties": {}
      }
    }
  ]
}
```

### 3.2 Tool Call Handler in johnny5.py

```python
from tool_executor import ToolExecutionEngine, RobotAdapter

# Initialize tool execution engine
robot_adapter = ChloeRobotAdapter()  # Our custom adapter
tool_engine = ToolExecutionEngine(robot_adapter)

async def on_message(message: SubscribeEvent, stream: Stream) -> None:
    global evi_last_activity, in_conversation

    # ... existing message handlers ...

    elif message.type == "tool_call":
        # Hume EVI is requesting a tool execution
        tool_name = message.name
        tool_call_id = message.tool_call_id
        arguments = json.loads(message.parameters)

        log(f"TOOL CALL: {tool_name}({arguments})")

        # LED: special color for tool execution?
        if LED_AVAILABLE:
            get_led().thinking()  # Or a new "executing" state

        try:
            # Execute the tool
            result = await tool_engine.execute_tool(
                ToolCall(name=tool_name, arguments=arguments)
            )

            # Send result back to Hume
            await socket.send_tool_response(
                tool_call_id=tool_call_id,
                content=json.dumps({
                    "success": result.success,
                    "message": result.message
                })
            )
            log(f"TOOL RESULT: {result}")

        except Exception as e:
            # Send error back to Hume
            await socket.send_tool_error(
                tool_call_id=tool_call_id,
                error=str(e),
                fallback_content=f"I tried to {tool_name} but encountered an error: {e}"
            )
            log(f"TOOL ERROR: {e}")

        # LED: back to listening
        if LED_AVAILABLE:
            get_led().listening()
```

### 3.3 Message Flow

```
User: "Chloe, wave at me"
         │
         ▼
┌─────────────────────────────────────────┐
│ Hume EVI processes speech               │
│ Infers tool: wave(arm="right")          │
│ Sends: tool_call message                │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ johnny5.py receives tool_call           │
│ Parses: name="wave", args={arm:"right"} │
│ Calls: tool_engine.execute_tool(...)    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ ToolExecutionEngine                     │
│ Creates action graph for "wave"         │
│ Executes: arm_to_wave_pose → wave_motion│
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ ChloeRobotAdapter                       │
│ Translates to Solo-CLI commands         │
│ Executes on hardware                    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ johnny5.py sends tool_response          │
│ {success: true, message: "Waved!"}      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Hume EVI generates spoken response      │
│ "There you go!" (with emotional tone)   │
└─────────────────────────────────────────┘
         │
         ▼
User hears: "There you go!" + sees wave
```

---

## Part 4: Robot Adapter Layer

### 4.1 Abstract Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class Subsystem(Enum):
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    BASE = "base"
    LIFT = "lift"
    GANTRY = "gantry"
    GRIPPER_LEFT = "gripper_left"
    GRIPPER_RIGHT = "gripper_right"

@dataclass
class ActionResult:
    success: bool
    message: str
    data: dict = None
    duration_ms: float = 0

class RobotAdapter(ABC):
    """Abstract interface for robot control.

    Implement this for each robot platform to enable
    the same tools to work across different hardware.
    """

    @abstractmethod
    async def execute(self, subsystem: Subsystem, action: ActionPrimitive) -> ActionResult:
        """Execute an action on a subsystem"""
        pass

    @abstractmethod
    async def get_state(self, subsystem: Subsystem) -> dict:
        """Get current state of a subsystem"""
        pass

    @abstractmethod
    async def stop(self, subsystem: Subsystem = None):
        """Emergency stop (all subsystems if None)"""
        pass

    @abstractmethod
    def get_capabilities(self) -> dict:
        """Return robot's capabilities for tool generation"""
        pass
```

### 4.2 Chloe-Specific Adapter

```python
import subprocess
import asyncio

class ChloeRobotAdapter(RobotAdapter):
    """Adapter for Chloe (Johnny Five) hardware"""

    def __init__(self):
        self.config = {
            "left_arm": {"port": "/dev/ttyACM0", "ids": [1,2,3,4,5,6]},
            "right_arm": {"port": "/dev/ttyACM1", "ids": [1,2,3,4,5,6]},
            "base": {"port": "/dev/ttyACM0", "ids": [7,8,9], "type": "mecanum"},
            "lift": {"port": "/dev/ttyACM0", "ids": [10]},
            "gantry": {"port": "/dev/ttyACM1", "ids": [7,8]},
        }

        # Named poses for each subsystem
        self.poses = {
            "left_arm": {
                "home": [0, -45, 90, 45, 0, 0],
                "wave": [0, -30, 120, 60, 0, 0],
                "point": [0, -60, 45, 90, 0, 0],
            },
            "right_arm": {
                "home": [0, -45, 90, 45, 0, 0],
                "wave": [0, -30, 120, 60, 0, 0],
                "point": [0, -60, 45, 90, 0, 0],
            },
            "gantry": {
                "center": [90, 90],
                "left": [45, 90],
                "right": [135, 90],
                "up": [90, 45],
                "down": [90, 135],
            }
        }

    async def execute(self, subsystem: Subsystem, action: ActionPrimitive) -> ActionResult:
        """Execute action using Solo-CLI or direct serial"""

        start_time = asyncio.get_event_loop().time()

        try:
            if action.name == "move_to_pose":
                result = await self._move_to_pose(subsystem, action.params)
            elif action.name == "move_to_position":
                result = await self._move_to_position(subsystem, action.params)
            elif action.name == "gripper":
                result = await self._gripper_action(subsystem, action.params)
            elif action.name == "base_move":
                result = await self._base_move(action.params)
            else:
                result = ActionResult(False, f"Unknown action: {action.name}")

            result.duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return result

        except Exception as e:
            return ActionResult(False, str(e))

    async def _move_to_pose(self, subsystem: Subsystem, params: dict) -> ActionResult:
        """Move subsystem to a named pose"""
        pose_name = params.get("pose")
        speed = params.get("speed", 0.5)

        if subsystem.value not in self.poses:
            return ActionResult(False, f"No poses defined for {subsystem}")

        if pose_name not in self.poses[subsystem.value]:
            return ActionResult(False, f"Unknown pose: {pose_name}")

        target = self.poses[subsystem.value][pose_name]

        # Call Solo-CLI
        cmd = self._build_solo_command(subsystem, target, speed)
        result = await self._run_command(cmd)

        return ActionResult(result.returncode == 0, result.stdout or result.stderr)

    def _build_solo_command(self, subsystem: Subsystem, positions: list, speed: float) -> list:
        """Build Solo-CLI command for movement"""
        config = self.config[subsystem.value]

        # Format: solo robo --port /dev/ttyACM0 --ids 1,2,3 --positions 0,45,90 --speed 0.5
        return [
            "solo", "robo",
            "--port", config["port"],
            "--ids", ",".join(map(str, config["ids"])),
            "--positions", ",".join(map(str, positions)),
            "--speed", str(speed)
        ]

    async def _run_command(self, cmd: list) -> subprocess.CompletedProcess:
        """Run a shell command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return subprocess.CompletedProcess(
            cmd, process.returncode,
            stdout.decode(), stderr.decode()
        )

    async def stop(self, subsystem: Subsystem = None):
        """Emergency stop"""
        if subsystem:
            # Stop specific subsystem
            cmd = ["solo", "robo", "--stop", "--port", self.config[subsystem.value]["port"]]
        else:
            # Stop all
            cmd = ["solo", "robo", "--stop", "--all"]

        await self._run_command(cmd)

    def get_capabilities(self) -> dict:
        """Return Chloe's capabilities"""
        return {
            "arms": ["left", "right"],
            "gripper": True,
            "mobile_base": True,
            "base_type": "mecanum",
            "lift": True,
            "gantry": True,
            "camera": "OAK-D",
        }
```

---

## Part 5: File Structure

```
Physical-Ai-Hack-2026/
├── johnny5.py                    # Main EVI integration (modify)
├── led_controller.py             # LED feedback (exists)
├── whoami_full.py                # Face recognition (exists)
│
├── tools/                        # NEW: Tool system
│   ├── __init__.py
│   ├── engine.py                 # ToolExecutionEngine
│   ├── graph.py                  # ActionGraph, dependency resolver
│   ├── primitives.py             # ActionPrimitive definitions
│   └── registry.py               # Tool registration for Hume
│
├── adapters/                     # NEW: Robot adapters
│   ├── __init__.py
│   ├── base.py                   # RobotAdapter ABC
│   ├── chloe.py                  # ChloeRobotAdapter
│   └── mock.py                   # MockRobotAdapter (for testing)
│
├── configs/                      # NEW: Robot configurations
│   ├── chloe.yaml                # Chloe's hardware config
│   ├── tools.json                # Hume EVI tool definitions
│   └── poses.yaml                # Named poses library
│
├── scripts/
│   ├── start_johnny5.sh          # Startup script (exists)
│   ├── enable_respeaker_output.sh # Audio config (exists)
│   └── calibrate_chloe.sh        # NEW: Calibration helper
│
└── forks/                        # NEW: Forked dependencies
    ├── chloe-solo/               # Forked solo-cli
    └── chloe-lerobot/            # Forked lerobot_alohamini
```

---

## Part 6: Implementation Timeline

### Day 1: Foundation (Today)

| Task | Time | Description |
|------|------|-------------|
| Fork repos | 30 min | Fork solo-cli and lerobot_alohamini |
| Create tool structure | 1 hr | Set up tools/, adapters/ directories |
| Define primitives | 1 hr | ActionPrimitive, ActionGraph classes |
| Mock adapter | 30 min | For testing without hardware |

### Day 2: Core Engine

| Task | Time | Description |
|------|------|-------------|
| Execution engine | 2 hr | Sequential/parallel execution |
| Dependency resolver | 1 hr | Topological sort, wave grouping |
| Chloe adapter | 2 hr | Solo-CLI integration |
| Basic tools | 1 hr | wave, look_at, stop |

### Day 3: Hume Integration

| Task | Time | Description |
|------|------|-------------|
| Tool definitions | 1 hr | JSON schema for Hume config |
| johnny5.py changes | 2 hr | tool_call handler, responses |
| Testing | 2 hr | End-to-end voice → action |
| Refinement | 1 hr | Error handling, timeouts |

### Day 4: Polish & Demo

| Task | Time | Description |
|------|------|-------------|
| Complex tools | 2 hr | pick_object, place_object |
| VLA integration | 2 hr | Connect to SmolVLA/ACT |
| Demo scenarios | 2 hr | Rehearse key demos |

---

## Part 7: Emergency Stop Architecture

**THE WINE EXAMPLE:** Robot is pouring wine. User says "STOP!" The robot MUST stop immediately. Yes, the wine will overflow and spill on the table. That's acceptable. What's NOT acceptable is the robot continuing to pour.

### 7.1 Emergency Stop Requirements

```
STOP must:
✓ Halt ALL motor movement within 50ms
✓ Bypass the tool execution engine entirely
✓ Override any pending actions
✓ Work even if main loop is blocked
✓ Be triggerable by voice, button, or software
✓ NOT wait for "graceful" completion
✓ NOT try to "finish the current action"

STOP must NOT:
✗ Wait for dependency resolution
✗ Queue behind other commands
✗ Check if it's "safe" to stop mid-action
✗ Try to be "smart" about it
```

### 7.2 Implementation: Separate Stop Thread

```python
import threading
import signal

class EmergencyStopSystem:
    """Hardware-level emergency stop that bypasses all software logic"""

    def __init__(self, serial_ports: list[str]):
        self.ports = serial_ports
        self.stopped = threading.Event()
        self._stop_thread = None

    def start(self):
        """Start listening for stop signals on separate thread"""
        self._stop_thread = threading.Thread(target=self._stop_listener, daemon=True)
        self._stop_thread.start()

    def trigger_stop(self):
        """IMMEDIATELY stop all motors - no questions asked"""
        self.stopped.set()

        # Direct serial write - bypass everything
        for port in self.ports:
            try:
                with serial.Serial(port, 1000000, timeout=0.01) as ser:
                    # Dynamixel broadcast torque disable
                    # ID 254 = broadcast to all motors
                    ser.write(self._build_torque_disable_packet(254))
            except:
                pass  # Try all ports even if some fail

        print("!!! EMERGENCY STOP TRIGGERED !!!")

    def _build_torque_disable_packet(self, motor_id: int) -> bytes:
        """Build Dynamixel protocol packet to disable torque"""
        # Protocol 2.0 torque disable
        # This is the fastest way to stop Dynamixel motors
        return bytes([
            0xFF, 0xFF, 0xFD, 0x00,  # Header
            motor_id,                 # ID (254 = broadcast)
            0x06, 0x00,               # Length
            0x03,                     # Write instruction
            0x40, 0x00,               # Torque Enable address
            0x00,                     # Value: 0 = disable
            0x00, 0x00                # CRC (simplified)
        ])

# Global stop system - initialized at startup
ESTOP = EmergencyStopSystem(["/dev/ttyACM0", "/dev/ttyACM1"])

# Voice trigger
async def on_message(message: SubscribeEvent, stream: Stream):
    # Check for stop BEFORE any other processing
    if message.type == "user_message":
        content = message.message.content.lower()
        if any(word in content for word in ["stop", "halt", "freeze", "emergency"]):
            ESTOP.trigger_stop()
            return  # Don't process anything else

# Signal handler for Ctrl+C
signal.signal(signal.SIGINT, lambda s, f: ESTOP.trigger_stop())
```

### 7.3 Asynchronous Safety Monitors

**THE OTHER WINE EXAMPLE:** Robot is pouring wine, AND dealing with an error on the left arm, AND someone just bumped it. The wine glass fill level sensor says "FULL". The pour MUST stop even though the main loop is busy handling other problems.

```python
class SafetyMonitor:
    """Independent async monitors that can interrupt ANY action"""

    def __init__(self, robot: RobotAdapter):
        self.robot = robot
        self.monitors: list[Callable] = []
        self.running = True
        self._monitor_task = None

    def register(self, condition: Callable, action: Callable, subsystems: list[str]):
        """Register a safety condition that triggers an action.

        condition: async () -> bool  (returns True when triggered)
        action: async () -> None     (what to do when triggered)
        subsystems: which subsystems this affects
        """
        self.monitors.append({
            "condition": condition,
            "action": action,
            "subsystems": subsystems
        })

    async def run(self):
        """Run all monitors continuously - INDEPENDENT of main loop"""
        while self.running:
            for monitor in self.monitors:
                try:
                    if await monitor["condition"]():
                        print(f"SAFETY MONITOR TRIGGERED: {monitor}")
                        await monitor["action"]()
                except Exception as e:
                    print(f"Safety monitor error: {e}")
            await asyncio.sleep(0.01)  # 10ms polling - fast enough for safety


# Example: Wine glass fill level monitor
async def wine_level_condition():
    """Check if wine glass is full - runs every 10ms regardless of what else is happening"""
    level = await sensors.get_wine_level()  # Could be vision, weight sensor, etc.
    return level >= WINE_GLASS_FULL_THRESHOLD

async def stop_pouring_action():
    """Stop the pour - this runs even if main loop is blocked"""
    await robot.execute(
        Subsystem.RIGHT_ARM,
        ActionPrimitive("stop_pour", params={"immediate": True})
    )

# Register it
safety = SafetyMonitor(robot)
safety.register(
    condition=wine_level_condition,
    action=stop_pouring_action,
    subsystems=["right_arm"]
)

# Start in background - NEVER blocks on main loop
asyncio.create_task(safety.run())
```

### 7.4 Condition-Based Action Interrupts

```python
@dataclass
class ActionPrimitive:
    name: str
    subsystem: str
    params: dict
    timeout: float
    dependencies: list[str]
    can_parallel: list[str]

    # NEW: Conditions that can interrupt this action mid-execution
    interrupt_conditions: list[Callable] = field(default_factory=list)
    on_interrupt: Callable = None  # What to do if interrupted


# Example: Pour wine with fill-level interrupt
pour_action = ActionPrimitive(
    name="pour_wine",
    subsystem="right_arm",
    params={"angle": 45, "duration": 5.0},
    timeout=10.0,
    dependencies=["position_over_glass"],
    can_parallel=[],
    interrupt_conditions=[
        lambda: sensors.wine_level() >= FULL,      # Glass full
        lambda: sensors.glass_removed(),            # Someone took the glass
        lambda: sensors.arm_force() > COLLISION,    # Hit something
    ],
    on_interrupt=lambda: robot.level_bottle()  # Return to upright
)
```

### 7.5 Independent Sensor Threads

```python
class SensorWatchdog:
    """Runs completely independently - even if asyncio event loop is blocked"""

    def __init__(self):
        self.callbacks: dict[str, list[Callable]] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)

    def on_condition(self, sensor: str, threshold: float, callback: Callable):
        """Register callback when sensor crosses threshold"""
        key = f"{sensor}:{threshold}"
        if key not in self.callbacks:
            self.callbacks[key] = []
        self.callbacks[key].append(callback)

    def _run(self):
        """Pure Python thread - no asyncio dependency"""
        while True:
            for key, callbacks in self.callbacks.items():
                sensor, threshold = key.split(":")
                value = self._read_sensor_sync(sensor)
                if value >= float(threshold):
                    for cb in callbacks:
                        try:
                            cb(value)
                        except:
                            pass
            time.sleep(0.01)  # 10ms

# Usage
watchdog = SensorWatchdog()
watchdog.on_condition("wine_level", 0.9, lambda v: ESTOP.stop_subsystem("right_arm"))
watchdog.on_condition("collision_force", 5.0, lambda v: ESTOP.trigger_stop())
watchdog.start()
```

### 7.6 Stop Priority Levels

| Level | Trigger | Action |
|-------|---------|--------|
| 0 - PANIC | Hardware button, "STOP NOW" | Immediate torque disable, power cut |
| 1 - EMERGENCY | Voice "stop", Ctrl+C | Immediate torque disable |
| 2 - ABORT | Tool error, timeout | Controlled deceleration, hold position |
| 3 - PAUSE | "Wait", "Hold on" | Stop after current primitive, hold position |
| 4 - CANCEL | "Never mind" | Complete current action, abort remaining |

---

## Part 8: Industry Examples & Existing Libraries

**Don't reinvent the wheel.** These are battle-tested:

### 8.1 Robot Control Frameworks

| Framework | Use For | Link |
|-----------|---------|------|
| **ROS 2 Actions** | Sequential execution with feedback | [ROS 2 Actions](https://docs.ros.org/en/rolling/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html) |
| **MoveIt Task Constructor** | Complex manipulation sequences | [MoveIt MTC](https://moveit.picknik.ai/main/doc/concepts/moveit_task_constructor/moveit_task_constructor.html) |
| **BehaviorTree.CPP** | Hierarchical task execution | [BT.CPP](https://www.behaviortree.dev/) |
| **PyRobot** | Facebook's robot control abstraction | [PyRobot](https://pyrobot.org/) |
| **Lerobot** | HuggingFace imitation learning | [LeRobot](https://github.com/huggingface/lerobot) |

### 8.2 Voice → Action Examples

| Project | Relevance | Link |
|---------|-----------|------|
| **Google PaLM-E** | Embodied language model | Paper |
| **RT-2** | Vision-Language-Action | Paper |
| **Language-to-Reward** | Voice → reward function | [L2R](https://language-to-reward.github.io/) |
| **SayCan** | Say what you can do | [SayCan](https://say-can.github.io/) |

### 8.3 Tool Calling Patterns

**OpenAI Function Calling** (Hume uses similar):
```python
# This pattern is well-established
tools = [
    {
        "type": "function",
        "function": {
            "name": "move_arm",
            "description": "Move robot arm",
            "parameters": {...}
        }
    }
]
```

**LangChain Tools** (for reference):
```python
from langchain.tools import tool

@tool
def move_arm(arm: str, position: str) -> str:
    """Move robot arm to position."""
    return robot.move(arm, position)
```

### 8.4 Existing Hume EVI Tool Examples

**From Hume's official examples:**
- [Weather Tool Example](https://github.com/HumeAI/hume-api-examples)
- [Function Calling Guide](https://dev.hume.ai/docs/speech-to-speech-evi/features/tool-use)

### 8.5 Parallel Execution Patterns

**asyncio.TaskGroup** (Python 3.11+):
```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(move_left_arm())
    tg.create_task(move_right_arm())
# Both complete before continuing
```

**trio** (alternative async library with better cancellation):
```python
async with trio.open_nursery() as nursery:
    nursery.start_soon(move_left_arm)
    nursery.start_soon(move_right_arm)
    # nursery.cancel_scope.cancel() stops both immediately
```

---

## Part 9: Risk Mitigation

### Risk: Hume EVI tool latency
**Mitigation:** Pre-load common tool patterns, use client-side intent detection for simple commands

### Risk: Motor communication failures
**Mitigation:** Retry logic, timeout handling, graceful degradation

### Risk: Dependency deadlocks
**Mitigation:** Cycle detection in action graph, timeout on dependency wait

### Risk: Safety during tool execution
**Mitigation:** Hardware limits in adapter, emergency stop always available, soft torque limits

---

## Part 8: Testing Strategy

### Unit Tests
```python
# test_graph.py
def test_dependency_resolution():
    graph = ActionGraph()
    graph.add_action(ActionPrimitive("A", deps=[]))
    graph.add_action(ActionPrimitive("B", deps=["A"]))
    graph.add_action(ActionPrimitive("C", deps=["A"]))
    graph.add_action(ActionPrimitive("D", deps=["B", "C"]))

    waves = graph.get_execution_order()
    assert waves == [["A"], ["B", "C"], ["D"]]
```

### Integration Tests
```python
# test_tools.py
async def test_wave_tool():
    adapter = MockRobotAdapter()
    engine = ToolExecutionEngine(adapter)

    result = await engine.execute_tool(
        ToolCall(name="wave", arguments={"arm": "right"})
    )

    assert result.success
    assert adapter.last_subsystem == Subsystem.RIGHT_ARM
```

### Hardware Tests
```bash
# Run on Jetson with real hardware
python -m pytest tests/hardware/ --robot=chloe
```

---

## References

- [Hume EVI Tool Use Documentation](https://dev.hume.ai/docs/speech-to-speech-evi/features/tool-use)
- [Hume Python SDK](https://github.com/HumeAI/hume-python-sdk)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [AlohaMini GitHub](https://github.com/liyiteng/AlohaMini)
- [LeRobot AlohaMini](https://github.com/liyiteng/lerobot_alohamini)
- [Solo CLI Documentation](https://docs.getsolo.tech/welcome)

---

## Next Steps

1. **Approve this plan** - Any changes to architecture or priorities?
2. **Fork repositories** - I can help set up the forks
3. **Start with mock adapter** - Test tool engine without hardware risk
4. **Iterate** - Get basic wave working, then expand

Ready to start implementing?
