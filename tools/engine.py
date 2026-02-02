"""Tool Execution Engine

This module provides the core execution engine that processes
tool calls from Hume EVI and executes them on the robot.

Features:
- Dependency resolution between actions
- Parallel execution of independent actions
- Emergency stop support
- Timeout handling
- Result aggregation
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import json

from adapters.base import RobotAdapter, Subsystem, ActionPrimitive, ActionResult


@dataclass
class ToolCall:
    """A tool call from Hume EVI."""
    name: str
    arguments: Dict[str, Any]
    tool_call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    message: str
    results: List[ActionResult] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convert to JSON for sending back to Hume."""
        return json.dumps({
            "success": self.success,
            "message": self.message,
            "data": self.data
        })


class ActionGraph:
    """Directed acyclic graph of robot actions for dependency resolution."""

    def __init__(self):
        self.actions: Dict[str, ActionPrimitive] = {}
        self.dependencies: Dict[str, Set[str]] = {}

    def add_action(self, action: ActionPrimitive) -> None:
        """Add an action to the graph."""
        self.actions[action.name] = action
        self.dependencies[action.name] = set(action.dependencies)

    def get_execution_waves(self) -> List[List[ActionPrimitive]]:
        """Get actions grouped by execution wave.

        Actions in the same wave can execute in parallel.
        Each wave must complete before the next begins.

        Returns:
            List of waves, where each wave is a list of actions
        """
        if not self.actions:
            return []

        completed: Set[str] = set()
        remaining = set(self.actions.keys())
        waves = []

        while remaining:
            # Find actions with all dependencies satisfied
            ready = {
                name for name in remaining
                if self.dependencies[name].issubset(completed)
            }

            if not ready:
                # Circular dependency detected
                raise ValueError(
                    f"Circular dependency detected. Remaining: {remaining}"
                )

            # Group ready actions by subsystem for parallel execution
            wave = [self.actions[name] for name in ready]
            waves.append(wave)

            completed.update(ready)
            remaining -= ready

        return waves


class ToolExecutionEngine:
    """Executes robot tools with dependency management.

    This engine receives tool calls from Hume EVI, parses them
    into action graphs, and executes them with proper ordering.
    """

    def __init__(self, robot: RobotAdapter):
        self.robot = robot
        self.tool_parsers: Dict[str, Callable] = {}
        self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """Register parsers for built-in tools."""
        self.tool_parsers["wave"] = self._parse_wave
        self.tool_parsers["move_arm"] = self._parse_move_arm
        self.tool_parsers["look_at"] = self._parse_look_at
        self.tool_parsers["pick_object"] = self._parse_pick_object
        self.tool_parsers["move_base"] = self._parse_move_base
        self.tool_parsers["stop"] = self._parse_stop
        self.tool_parsers["go_to_pose"] = self._parse_go_to_pose

    def register_parser(
        self, tool_name: str, parser: Callable[[Dict], ActionGraph]
    ) -> None:
        """Register a custom tool parser."""
        self.tool_parsers[tool_name] = parser

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call from Hume EVI.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with success status and details
        """
        # Check for emergency stop
        if self.robot.is_stopped:
            return ToolResult(
                success=False,
                message="Robot is in emergency stop state. Reset required."
            )

        # Handle stop command specially
        if tool_call.name == "stop":
            await self.robot.stop()
            return ToolResult(
                success=True,
                message="Emergency stop activated"
            )

        # Parse tool into action graph
        try:
            graph = self._parse_tool(tool_call)
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Failed to parse tool: {e}"
            )

        # Execute action waves
        all_results = []
        try:
            waves = graph.get_execution_waves()

            for wave in waves:
                wave_results = await self._execute_wave(wave)
                all_results.extend(wave_results)

                # Check for failures
                if any(r.failed for r in wave_results):
                    failed = [r for r in wave_results if r.failed]
                    return ToolResult(
                        success=False,
                        message=f"Action failed: {failed[0].message}",
                        results=all_results
                    )

        except asyncio.CancelledError:
            return ToolResult(
                success=False,
                message="Execution cancelled",
                results=all_results
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Execution error: {e}",
                results=all_results
            )

        return ToolResult(
            success=True,
            message=f"Completed {tool_call.name}",
            results=all_results
        )

    def _parse_tool(self, tool_call: ToolCall) -> ActionGraph:
        """Parse a tool call into an action graph."""
        parser = self.tool_parsers.get(tool_call.name)
        if not parser:
            raise ValueError(f"Unknown tool: {tool_call.name}")
        return parser(tool_call.arguments)

    async def _execute_wave(
        self, actions: List[ActionPrimitive]
    ) -> List[ActionResult]:
        """Execute a wave of actions (can be parallel)."""

        # Group by subsystem - same subsystem must be sequential
        by_subsystem: Dict[str, List[ActionPrimitive]] = defaultdict(list)
        for action in actions:
            by_subsystem[action.subsystem].append(action)

        # Execute each subsystem group in parallel
        tasks = []
        for subsystem_name, subsystem_actions in by_subsystem.items():
            subsystem = Subsystem(subsystem_name)
            task = asyncio.create_task(
                self._execute_subsystem_actions(subsystem, subsystem_actions)
            )
            tasks.append(task)

        # Wait for all subsystems to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        flat_results = []
        for result in results:
            if isinstance(result, Exception):
                flat_results.append(ActionResult(
                    success=False,
                    message=str(result)
                ))
            elif isinstance(result, list):
                flat_results.extend(result)
            else:
                flat_results.append(result)

        return flat_results

    async def _execute_subsystem_actions(
        self, subsystem: Subsystem, actions: List[ActionPrimitive]
    ) -> List[ActionResult]:
        """Execute actions on a single subsystem (sequential)."""
        results = []
        for action in actions:
            try:
                result = await asyncio.wait_for(
                    self.robot.execute(subsystem, action),
                    timeout=action.timeout
                )
            except asyncio.TimeoutError:
                result = ActionResult(
                    success=False,
                    message=f"Action '{action.name}' timed out after {action.timeout}s",
                    subsystem=subsystem
                )

            results.append(result)
            if result.failed:
                break  # Stop on first failure

        return results

    # =========================================================================
    # Tool Parsers
    # =========================================================================

    def _parse_wave(self, args: Dict) -> ActionGraph:
        """Parse wave tool into actions."""
        graph = ActionGraph()
        arm = args.get("arm", "right")
        style = args.get("style", "friendly")

        subsystem = "right_arm" if arm == "right" else "left_arm"

        graph.add_action(ActionPrimitive(
            name="wave",
            subsystem=subsystem,
            params={"style": style, "arm": arm},
            timeout=10.0
        ))

        return graph

    def _parse_move_arm(self, args: Dict) -> ActionGraph:
        """Parse move_arm tool into actions."""
        graph = ActionGraph()
        arm = args.get("arm", "right")
        position = args.get("position", "home")
        speed = args.get("speed", 0.5)

        subsystem = "right_arm" if arm == "right" else "left_arm"

        graph.add_action(ActionPrimitive(
            name="move_to_pose" if isinstance(position, str) else "move_to_position",
            subsystem=subsystem,
            params={"pose": position, "speed": speed} if isinstance(position, str)
                   else {"positions": position, "speed": speed},
            timeout=15.0
        ))

        return graph

    def _parse_look_at(self, args: Dict) -> ActionGraph:
        """Parse look_at tool into actions."""
        graph = ActionGraph()
        target = args.get("target", "forward")

        graph.add_action(ActionPrimitive(
            name="look_at",
            subsystem="gantry",
            params={"target": target},
            timeout=5.0
        ))

        return graph

    def _parse_pick_object(self, args: Dict) -> ActionGraph:
        """Parse pick_object tool into a sequence of actions."""
        graph = ActionGraph()
        obj = args.get("object", "object")
        hand = args.get("hand", "right")

        subsystem = "right_arm" if hand == "right" else "left_arm"

        # Look at object first
        graph.add_action(ActionPrimitive(
            name="look_at",
            subsystem="gantry",
            params={"target": obj},
            timeout=5.0,
            dependencies=[]
        ))

        # Move arm to object
        graph.add_action(ActionPrimitive(
            name="move_to_position",
            subsystem=subsystem,
            params={"target": obj, "offset": [0, 0, 0.1]},
            timeout=10.0,
            dependencies=["look_at"]
        ))

        # Open gripper
        graph.add_action(ActionPrimitive(
            name="gripper",
            subsystem=subsystem,
            params={"action": "open"},
            timeout=3.0,
            dependencies=["move_to_position"]
        ))

        # Lower to object
        graph.add_action(ActionPrimitive(
            name="approach",
            subsystem=subsystem,
            params={"target": obj},
            timeout=5.0,
            dependencies=["gripper"]
        ))

        # Close gripper
        graph.add_action(ActionPrimitive(
            name="grasp",
            subsystem=subsystem,
            params={"action": "close", "force": 0.5},
            timeout=3.0,
            dependencies=["approach"]
        ))

        # Lift
        graph.add_action(ActionPrimitive(
            name="lift",
            subsystem=subsystem,
            params={"height": 0.15},
            timeout=5.0,
            dependencies=["grasp"]
        ))

        return graph

    def _parse_move_base(self, args: Dict) -> ActionGraph:
        """Parse move_base tool into actions."""
        graph = ActionGraph()
        direction = args.get("direction", "forward")
        distance = args.get("distance", 0.5)

        graph.add_action(ActionPrimitive(
            name="base_move",
            subsystem="base",
            params={"direction": direction, "distance": distance},
            timeout=10.0
        ))

        return graph

    def _parse_stop(self, args: Dict) -> ActionGraph:
        """Parse stop tool - returns empty graph, handled specially."""
        return ActionGraph()

    def _parse_go_to_pose(self, args: Dict) -> ActionGraph:
        """Parse go_to_pose tool."""
        graph = ActionGraph()
        pose = args.get("pose", "home")

        # Add action for each subsystem in the pose
        graph.add_action(ActionPrimitive(
            name="move_to_pose",
            subsystem="all",
            params={"pose": pose},
            timeout=15.0
        ))

        return graph
