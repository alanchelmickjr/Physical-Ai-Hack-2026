# Hume EVI Tool Calls

This document describes how Hume EVI tool calls work with the robot, including the tool registry, execution engine, and parallel/serial execution with dependencies.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HUME EVI (Cloud)                                                        │
│  "Wave at Alan" → tool_call: wave(arm="right")                          │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ WebSocket
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TOOL REGISTRY (tools/registry.py)                                       │
│  50+ tool definitions with JSON schemas                                  │
│  - Movement: wave, look_at, move_arm, pick_object                       │
│  - Expression: express, nod, gesture_while_speaking                     │
│  - Locomotion: walk, spin, turn, dance                                  │
│  - Safety: alert_hazard, enable_fire_detection                          │
│  - Calibration: calibrate_arm, setup_robot, self_test                   │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TOOL EXECUTION ENGINE (tools/engine.py)                                 │
│  - Parses tool calls into ActionGraph                                    │
│  - Resolves dependencies (DAG)                                          │
│  - Groups actions into execution waves                                   │
│  - Executes waves in parallel where possible                            │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ROBOT ADAPTER (adapters/base.py)                                        │
│  - Johnny Five: Solo-CLI / Feetech servos                               │
│  - Booster K1: ROS2 topics / high-level APIs                            │
│  - Unitree G1: unitree_sdk2 / ROS2                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tool Registry

The tool registry (`tools/registry.py`) defines 50+ tools that Hume EVI can call.

### Tool Definition Format

```python
ToolDefinition(
    name="wave",
    description="Wave hello or goodbye with robot arm",
    parameters={
        "type": "object",
        "properties": {
            "arm": {
                "type": "string",
                "enum": ["left", "right", "both"],
                "default": "right",
                "description": "Which arm to wave with"
            },
            "style": {
                "type": "string",
                "enum": ["friendly", "excited", "royal", "shy"],
                "default": "friendly",
                "description": "Style of wave"
            }
        }
    }
)
```

### Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| **Movement** | `wave`, `move_arm`, `look_at`, `nod` | Basic movement primitives |
| **Expression** | `express`, `gesture_while_speaking` | Emotional body language |
| **Manipulation** | `pick_object`, `place_object`, `gripper` | Object interaction |
| **Locomotion** | `move_base`, `walk`, `spin`, `turn`, `dance` | Mobile base control |
| **Safety** | `stop`, `alert_hazard`, `enable_fire_detection` | Safety systems |
| **Lift/Hitch** | `lift`, `hitch`, `tow_cart`, `dock_charger` | Vertical/grabber |
| **Diagnostics** | `scan_motors`, `motor_status`, `check_torque` | Hardware monitoring |
| **Calibration** | `calibrate_arm`, `setup_robot`, `self_test` | Setup/configuration |

### Exporting for Hume

```python
from tools.registry import get_registry

registry = get_registry()

# Export as JSON for Hume config
tools_json = registry.to_json()

# Export as Hume format (list of dicts)
tools_list = registry.to_hume_format()
```

---

## Execution Engine

The execution engine (`tools/engine.py`) processes tool calls with dependency management.

### ActionGraph

The `ActionGraph` class organizes actions into a DAG (directed acyclic graph) for dependency resolution.

```python
class ActionGraph:
    def add_action(self, action: ActionPrimitive) -> None
    def get_execution_waves(self) -> List[List[ActionPrimitive]]
```

### Execution Waves

Actions are grouped into "waves" based on dependencies:
- **Wave 1**: Actions with no dependencies (run in parallel)
- **Wave 2**: Actions depending on Wave 1 (run after Wave 1)
- **Wave N**: Actions depending on Wave N-1

```
Wave 1: [look_at, open_gripper]     ← Run in parallel
         │         │
         ▼         ▼
Wave 2: [move_arm_to_object]        ← Depends on look_at
         │
         ▼
Wave 3: [close_gripper]             ← Depends on move_arm
         │
         ▼
Wave 4: [lift_object]               ← Depends on close_gripper
```

### Parallel vs Serial Execution

| Scenario | Execution |
|----------|-----------|
| Different subsystems, no dependencies | **Parallel** |
| Same subsystem | **Serial** (queued) |
| Has dependencies | **Serial** (waits for deps) |
| Different subsystems with dependencies | **Parallel within wave** |

### Example: pick_object

```python
def _parse_pick_object(self, args: Dict) -> ActionGraph:
    graph = ActionGraph()

    # 1. Look at object (no deps)
    graph.add_action(ActionPrimitive(
        name="look_at",
        subsystem="gantry",
        params={"target": obj},
        dependencies=[]
    ))

    # 2. Move arm to object (depends on look_at)
    graph.add_action(ActionPrimitive(
        name="move_to_position",
        subsystem=arm_subsystem,
        params={"target": obj},
        dependencies=["look_at"]
    ))

    # 3. Open gripper (depends on move_to_position)
    graph.add_action(ActionPrimitive(
        name="gripper",
        subsystem=arm_subsystem,
        params={"action": "open"},
        dependencies=["move_to_position"]
    ))

    # 4. Approach (depends on gripper open)
    graph.add_action(ActionPrimitive(
        name="approach",
        subsystem=arm_subsystem,
        params={"target": obj},
        dependencies=["gripper"]
    ))

    # 5. Grasp (depends on approach)
    graph.add_action(ActionPrimitive(
        name="grasp",
        subsystem=arm_subsystem,
        params={"action": "close"},
        dependencies=["approach"]
    ))

    # 6. Lift (depends on grasp)
    graph.add_action(ActionPrimitive(
        name="lift",
        subsystem=arm_subsystem,
        params={"height": 0.15},
        dependencies=["grasp"]
    ))

    return graph
```

Execution waves for this:
```
Wave 1: [look_at]
Wave 2: [move_to_position]
Wave 3: [gripper:open]
Wave 4: [approach]
Wave 5: [grasp]
Wave 6: [lift]
```

---

## Parallel Execution Example

For the `setup_robot` tool with multiple calibration steps:

```python
def _parse_setup_robot(self, args: Dict) -> ActionGraph:
    graph = ActionGraph()

    # Wave 1: Scan (no deps)
    graph.add_action(ActionPrimitive(
        name="scan_motors",
        subsystem="all",
        dependencies=[]
    ))

    # Wave 2: Calibrate arms (both in parallel, depend on scan)
    graph.add_action(ActionPrimitive(
        name="calibrate",
        subsystem="left_arm",
        dependencies=["scan_motors"]
    ))
    graph.add_action(ActionPrimitive(
        name="calibrate",
        subsystem="right_arm",
        dependencies=["scan_motors"]
    ))

    # Wave 3: Calibrate gantry (depends on arms)
    graph.add_action(ActionPrimitive(
        name="calibrate_gantry",
        subsystem="gantry",
        dependencies=["calibrate"]  # Waits for both arms
    ))

    return graph
```

Execution:
```
Wave 1: [scan_motors]              ← Serial (single action)
Wave 2: [calibrate:left_arm, calibrate:right_arm]  ← PARALLEL!
Wave 3: [calibrate_gantry]         ← Serial (waits for Wave 2)
```

---

## Adding Custom Tools

### 1. Register in registry.py

```python
self.register(ToolDefinition(
    name="dance",
    description="Do a little dance movement",
    parameters={
        "type": "object",
        "properties": {
            "style": {
                "type": "string",
                "enum": ["happy", "silly", "groovy", "victory"],
                "default": "happy"
            },
            "duration": {
                "type": "number",
                "default": 3.0
            }
        }
    }
))
```

### 2. Add parser in engine.py

```python
def _parse_dance(self, args: Dict) -> ActionGraph:
    graph = ActionGraph()
    style = args.get("style", "happy")
    duration = args.get("duration", 3.0)

    # Dance uses both arms in parallel
    graph.add_action(ActionPrimitive(
        name="dance_sequence",
        subsystem="left_arm",
        params={"style": style, "duration": duration},
        dependencies=[]
    ))
    graph.add_action(ActionPrimitive(
        name="dance_sequence",
        subsystem="right_arm",
        params={"style": style, "duration": duration},
        dependencies=[]
    ))

    return graph
```

### 3. Register parser

```python
self.tool_parsers["dance"] = self._parse_dance
```

---

## Booster K1 Integration

The Booster K1 exposes high-level ROS2 APIs that map well to tools:

### ROS2 Topics

| Topic | Type | Purpose |
|-------|------|---------|
| `/cmd_vel` | `geometry_msgs/Twist` | Base velocity |
| `/joint_states` | `sensor_msgs/JointState` | Joint positions |
| `/head/pan_tilt` | `std_msgs/Float64MultiArray` | Head control |
| `/arm/joint_positions` | `std_msgs/Float64MultiArray` | Arm control |

### Named Poses

```python
BOOSTER_K1_POSES = {
    "home": {LEFT_ARM: [0,0,0,0,0,0], RIGHT_ARM: [0,0,0,0,0,0], GANTRY: [0,0]},
    "wave": {RIGHT_ARM: [-30, 45, 0, 90, 0, 0]},
    "wave_left": {LEFT_ARM: [-30, -45, 0, 90, 0, 0]},
    "arms_open": {LEFT_ARM: [0,-60,0,0,0,0], RIGHT_ARM: [0,60,0,0,0,0]},
    "look_down": {GANTRY: [0, 30]},
    "look_up": {GANTRY: [0, -20]},
}
```

### Dance Sequence Example

For the Booster K1, a dance might be:

```python
async def dance_happy(self, duration: float = 3.0):
    """Happy dance - arms up, wave side to side."""
    start = time.time()

    while time.time() - start < duration:
        # Arms up
        await self.move_to_pose("arms_open")
        await asyncio.sleep(0.3)

        # Wave left
        await self._move_to_position(Subsystem.LEFT_ARM,
            {"positions": [-30, -60, 0, 60, 0, 0], "speed": 0.8})
        await self._move_to_position(Subsystem.RIGHT_ARM,
            {"positions": [-30, 60, 0, 60, 0, 0], "speed": 0.8})
        await asyncio.sleep(0.3)

        # Wave right
        await self._move_to_position(Subsystem.LEFT_ARM,
            {"positions": [30, -60, 0, 60, 0, 0], "speed": 0.8})
        await self._move_to_position(Subsystem.RIGHT_ARM,
            {"positions": [30, 60, 0, 60, 0, 0], "speed": 0.8})
        await asyncio.sleep(0.3)

    # Return home
    await self.move_to_pose("home")
```

---

## Error Handling

### Timeout

```python
async def _execute_subsystem_actions(self, subsystem, actions):
    for action in actions:
        try:
            result = await asyncio.wait_for(
                self.robot.execute(subsystem, action),
                timeout=action.timeout
            )
        except asyncio.TimeoutError:
            result = ActionResult(
                success=False,
                message=f"Action '{action.name}' timed out"
            )
```

### Emergency Stop

```python
async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
    if self.robot.is_stopped:
        return ToolResult(
            success=False,
            message="Robot is in emergency stop state"
        )

    if tool_call.name == "stop":
        await self.robot.stop()
        return ToolResult(success=True, message="Emergency stop activated")
```

### Failure Propagation

If any action in a wave fails:
1. Stop that subsystem's remaining actions
2. Complete other subsystems in the wave
3. Return failure result with partial completion info

---

## Tool Result Format

Results are sent back to Hume in JSON:

```json
{
    "success": true,
    "message": "Completed wave",
    "data": {
        "duration_ms": 1523,
        "subsystems": ["right_arm", "gantry"]
    }
}
```

---

## Best Practices

1. **Keep tools atomic** - One tool = one intent
2. **Use dependencies** - Let the engine parallelize
3. **Set appropriate timeouts** - Arm moves: 15s, calibration: 60s
4. **Handle graceful degradation** - Check robot capabilities first
5. **Log tool calls** - For debugging and replays

---

## Testing Tools

```bash
# Test tool parsing
python -c "
from tools.engine import ToolExecutionEngine, ToolCall
from adapters.johnny5 import Johnny5Adapter

robot = Johnny5Adapter()
engine = ToolExecutionEngine(robot)

call = ToolCall(name='wave', arguments={'arm': 'right', 'style': 'friendly'})
graph = engine._parse_tool(call)
waves = graph.get_execution_waves()
print(f'Waves: {len(waves)}')
for i, wave in enumerate(waves):
    print(f'  Wave {i+1}: {[a.name for a in wave]}')
"
```
