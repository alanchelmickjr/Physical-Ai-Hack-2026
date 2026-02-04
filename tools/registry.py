"""Tool Registry for Hume EVI

This module provides a registry of tools that can be called by Hume EVI.
Tools are defined with JSON schemas for automatic integration.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json
import functools


@dataclass
class ToolDefinition:
    """Definition of a tool for Hume EVI."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None


class ToolRegistry:
    """Registry of tools available to Hume EVI."""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_builtin_tools()

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)

    def to_hume_format(self) -> List[Dict]:
        """Export tools in Hume EVI configuration format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]

    def to_json(self) -> str:
        """Export tools as JSON."""
        return json.dumps(self.to_hume_format(), indent=2)

    def _register_builtin_tools(self) -> None:
        """Register built-in robot control tools."""

        self.register(ToolDefinition(
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
        ))

        self.register(ToolDefinition(
            name="move_arm",
            description="Move robot arm to a position or named pose",
            parameters={
                "type": "object",
                "properties": {
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right", "both"],
                        "description": "Which arm to move"
                    },
                    "position": {
                        "type": "string",
                        "description": "Named pose (home, wave, point, ready) or coordinates"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.5,
                        "description": "Movement speed"
                    }
                },
                "required": ["arm", "position"]
            }
        ))

        self.register(ToolDefinition(
            name="look_at",
            description="Point the camera/head to look at something",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "What to look at: person name, object, or direction (left, right, up, down, forward)"
                    }
                },
                "required": ["target"]
            }
        ))

        self.register(ToolDefinition(
            name="pick_object",
            description="Pick up an object that the robot can see",
            parameters={
                "type": "object",
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "Description of the object to pick up"
                    },
                    "hand": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "default": "right",
                        "description": "Which hand to use"
                    }
                },
                "required": ["object"]
            }
        ))

        self.register(ToolDefinition(
            name="place_object",
            description="Place a held object somewhere",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Where to place the object"
                    },
                    "hand": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "default": "right"
                    }
                },
                "required": ["location"]
            }
        ))

        self.register(ToolDefinition(
            name="move_base",
            description="Move the robot's mobile base",
            parameters={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "left", "right", "rotate_left", "rotate_right"],
                        "description": "Direction to move"
                    },
                    "distance": {
                        "type": "number",
                        "description": "Distance in meters (or degrees for rotation)",
                        "default": 0.5
                    }
                },
                "required": ["direction"]
            }
        ))

        self.register(ToolDefinition(
            name="stop",
            description="Immediately stop all robot movement",
            parameters={
                "type": "object",
                "properties": {}
            }
        ))

        # =====================================================================
        # EXTENDED MOVEMENT COMMANDS
        # =====================================================================

        self.register(ToolDefinition(
            name="spin",
            description="Spin the robot in place (full 360 or partial)",
            parameters={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["clockwise", "counterclockwise", "cw", "ccw"],
                        "description": "Spin direction"
                    },
                    "degrees": {
                        "type": "number",
                        "default": 360,
                        "description": "Degrees to spin (default full rotation)"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.5,
                        "description": "Spin speed"
                    }
                },
                "required": ["direction"]
            }
        ))

        self.register(ToolDefinition(
            name="turn",
            description="Turn to face a direction or angle",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Direction (left, right, around, behind) or angle in degrees"
                    },
                    "speed": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "default": 0.5,
                        "description": "Turn speed"
                    }
                },
                "required": ["target"]
            }
        ))

        self.register(ToolDefinition(
            name="dance",
            description="Do a little dance movement",
            parameters={
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "enum": ["happy", "silly", "groovy", "victory"],
                        "default": "happy",
                        "description": "Dance style"
                    },
                    "duration": {
                        "type": "number",
                        "default": 3.0,
                        "description": "How long to dance (seconds)"
                    }
                }
            }
        ))

        # =====================================================================
        # EXPRESSIVE MOVEMENT (arms during speech)
        # =====================================================================

        self.register(ToolDefinition(
            name="express",
            description="Express an emotion through body language",
            parameters={
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": ["happy", "sad", "excited", "curious", "thinking", "surprised", "confused", "proud", "shy"],
                        "description": "Emotion to express"
                    },
                    "intensity": {
                        "type": "string",
                        "enum": ["subtle", "normal", "dramatic"],
                        "default": "normal",
                        "description": "How pronounced the expression"
                    }
                },
                "required": ["emotion"]
            }
        ))

        self.register(ToolDefinition(
            name="point_at_person",
            description="Point at a person by name (rate-limited: once per 2 minutes per person to avoid being rude)",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the person to point at"
                    },
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right", "auto"],
                        "default": "auto",
                        "description": "Which arm to use (auto picks based on location)"
                    },
                    "context": {
                        "type": "string",
                        "enum": ["introduction", "clarification", "emphasis"],
                        "default": "introduction",
                        "description": "Why pointing (affects rate limiting)"
                    }
                },
                "required": ["name"]
            }
        ))

        self.register(ToolDefinition(
            name="nod",
            description="Nod head yes or shake head no",
            parameters={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["yes", "no", "maybe"],
                        "default": "yes",
                        "description": "Type of head movement"
                    },
                    "intensity": {
                        "type": "string",
                        "enum": ["subtle", "normal", "emphatic"],
                        "default": "normal"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="gesture_while_speaking",
            description="Enable/disable automatic arm gestures during speech",
            parameters={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to gesture while speaking"
                    },
                    "style": {
                        "type": "string",
                        "enum": ["minimal", "natural", "expressive", "dramatic"],
                        "default": "natural",
                        "description": "Gesture style"
                    }
                }
            }
        ))

        # =====================================================================
        # FAULT TOLERANCE & RECOVERY
        # =====================================================================

        self.register(ToolDefinition(
            name="check_servos",
            description="Check which servos are responding and report any missing",
            parameters={
                "type": "object",
                "properties": {
                    "subsystem": {
                        "type": "string",
                        "enum": ["all", "left_arm", "right_arm", "base", "lift", "gantry", "hitch"],
                        "default": "all",
                        "description": "Which subsystem to check"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="recover_servo",
            description="Attempt to recover a non-responding servo",
            parameters={
                "type": "object",
                "properties": {
                    "bus": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which bus"
                    },
                    "motor_id": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 11,
                        "description": "Motor ID to recover"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["ping", "reboot", "reset_error", "reconfigure"],
                        "default": "ping",
                        "description": "Recovery action to try"
                    }
                },
                "required": ["bus", "motor_id"]
            }
        ))

        self.register(ToolDefinition(
            name="enable_degraded_mode",
            description="Enable operation with missing servos (work around faults)",
            parameters={
                "type": "object",
                "properties": {
                    "missing_servos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of missing servo identifiers (e.g., ['left_arm_3', 'gantry_tilt'])"
                    },
                    "continue_anyway": {
                        "type": "boolean",
                        "default": True,
                        "description": "Continue operation despite missing servos"
                    }
                },
                "required": ["missing_servos"]
            }
        ))

        self.register(ToolDefinition(
            name="servo_health_report",
            description="Get full health report on all servos",
            parameters={
                "type": "object",
                "properties": {
                    "include_temps": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include temperature readings"
                    },
                    "include_loads": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include load/torque readings"
                    }
                }
            }
        ))

        # =====================================================================
        # SAFETY & HAZARD DETECTION
        # =====================================================================

        self.register(ToolDefinition(
            name="enable_fire_detection",
            description="Enable visual fire/smoke detection (OAK-D camera)",
            parameters={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable or disable fire detection"
                    },
                    "sensitivity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "default": "medium",
                        "description": "Detection sensitivity"
                    },
                    "auto_alert": {
                        "type": "boolean",
                        "default": True,
                        "description": "Automatically point at detected fire"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="alert_hazard",
            description="Alert to a detected hazard (point at it, speak warning)",
            parameters={
                "type": "object",
                "properties": {
                    "hazard_type": {
                        "type": "string",
                        "enum": ["fire", "smoke", "obstacle", "person_down", "spill", "unknown"],
                        "description": "Type of hazard detected"
                    },
                    "direction_degrees": {
                        "type": "number",
                        "description": "Direction of hazard (DOA angle)"
                    },
                    "speak_warning": {
                        "type": "boolean",
                        "default": True,
                        "description": "Speak a warning about the hazard"
                    }
                },
                "required": ["hazard_type"]
            }
        ))

        self.register(ToolDefinition(
            name="go_to_pose",
            description="Move the entire robot to a named pose",
            parameters={
                "type": "object",
                "properties": {
                    "pose": {
                        "type": "string",
                        "enum": ["home", "ready", "wave", "point", "arms_up", "arms_down"],
                        "description": "Name of the pose"
                    }
                },
                "required": ["pose"]
            }
        ))

        self.register(ToolDefinition(
            name="gripper",
            description="Control the gripper on an arm",
            parameters={
                "type": "object",
                "properties": {
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "default": "right"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["open", "close", "toggle"],
                        "default": "toggle"
                    }
                }
            }
        ))

        # =====================================================================
        # DIAGNOSTIC TOOLS
        # =====================================================================

        self.register(ToolDefinition(
            name="scan_motors",
            description="Scan for connected motors on a bus and report their status",
            parameters={
                "type": "object",
                "properties": {
                    "bus": {
                        "type": "string",
                        "enum": ["left", "right", "both"],
                        "default": "both",
                        "description": "Which bus to scan (left=ACM0, right=ACM1)"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="motor_status",
            description="Get detailed status of a specific motor or subsystem",
            parameters={
                "type": "object",
                "properties": {
                    "subsystem": {
                        "type": "string",
                        "enum": ["left_arm", "right_arm", "gantry", "lift", "base", "all"],
                        "description": "Which subsystem to check"
                    },
                    "motor_id": {
                        "type": "integer",
                        "description": "Specific motor ID (optional, checks all in subsystem if not specified)"
                    }
                },
                "required": ["subsystem"]
            }
        ))

        self.register(ToolDefinition(
            name="test_motor",
            description="Test a single motor by moving it slightly and back",
            parameters={
                "type": "object",
                "properties": {
                    "bus": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which bus (left=ACM0, right=ACM1)"
                    },
                    "motor_id": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Motor ID to test"
                    },
                    "angle": {
                        "type": "number",
                        "default": 10,
                        "description": "Degrees to move for test"
                    }
                },
                "required": ["bus", "motor_id"]
            }
        ))

        self.register(ToolDefinition(
            name="check_torque",
            description="Check if motors have torque enabled",
            parameters={
                "type": "object",
                "properties": {
                    "subsystem": {
                        "type": "string",
                        "enum": ["left_arm", "right_arm", "gantry", "lift", "base", "all"],
                        "description": "Which subsystem to check"
                    }
                },
                "required": ["subsystem"]
            }
        ))

        self.register(ToolDefinition(
            name="enable_torque",
            description="Enable or disable torque on motors",
            parameters={
                "type": "object",
                "properties": {
                    "subsystem": {
                        "type": "string",
                        "enum": ["left_arm", "right_arm", "gantry", "lift", "base", "all"],
                        "description": "Which subsystem"
                    },
                    "enable": {
                        "type": "boolean",
                        "default": True,
                        "description": "True to enable, False to disable (limp mode)"
                    }
                },
                "required": ["subsystem"]
            }
        ))

        # =====================================================================
        # SETUP & CALIBRATION TOOLS
        # =====================================================================

        self.register(ToolDefinition(
            name="calibrate_arm",
            description="Run calibration sequence for an SO101 arm",
            parameters={
                "type": "object",
                "properties": {
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which arm to calibrate"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["full", "quick", "offsets_only"],
                        "default": "quick",
                        "description": "Calibration mode"
                    }
                },
                "required": ["arm"]
            }
        ))

        self.register(ToolDefinition(
            name="calibrate_gantry",
            description="Calibrate the camera gantry (pan/tilt)",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["full", "center_only"],
                        "default": "center_only",
                        "description": "Calibration mode"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="calibrate_lift",
            description="Calibrate the 30cm lift mechanism",
            parameters={
                "type": "object",
                "properties": {
                    "find_limits": {
                        "type": "boolean",
                        "default": True,
                        "description": "Find top and bottom limits"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="calibrate_base",
            description="Calibrate the omni wheel base",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["wheel_test", "drive_test", "full"],
                        "default": "wheel_test",
                        "description": "Calibration mode"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="set_motor_id",
            description="Change a motor's ID (use during initial setup)",
            parameters={
                "type": "object",
                "properties": {
                    "bus": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which bus"
                    },
                    "current_id": {
                        "type": "integer",
                        "description": "Current motor ID"
                    },
                    "new_id": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 253,
                        "description": "New motor ID"
                    }
                },
                "required": ["bus", "current_id", "new_id"]
            }
        ))

        self.register(ToolDefinition(
            name="set_home_position",
            description="Set current position as home for a subsystem",
            parameters={
                "type": "object",
                "properties": {
                    "subsystem": {
                        "type": "string",
                        "enum": ["left_arm", "right_arm", "gantry", "lift"],
                        "description": "Which subsystem"
                    }
                },
                "required": ["subsystem"]
            }
        ))

        # =====================================================================
        # LIFT CONTROL
        # =====================================================================

        self.register(ToolDefinition(
            name="lift",
            description="Control the 30cm lift mechanism",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["up", "down", "home", "top", "bottom"],
                        "description": "Lift action"
                    },
                    "distance_mm": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 300,
                        "description": "Distance in mm (for up/down)"
                    },
                    "position_mm": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 300,
                        "description": "Absolute position in mm"
                    }
                },
                "required": ["action"]
            }
        ))

        # =====================================================================
        # HITCH CONTROL (Rear Grabber)
        # =====================================================================

        self.register(ToolDefinition(
            name="hitch",
            description="Control the rear hitch/grabber for cart towing or charger docking",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open", "close", "toggle", "dock", "release"],
                        "description": "Hitch action (dock=close+verify contact, release=open)"
                    },
                    "grip_percent": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Grip percentage (0=open, 100=closed)"
                    }
                },
                "required": ["action"]
            }
        ))

        self.register(ToolDefinition(
            name="calibrate_hitch",
            description="Calibrate the rear hitch/grabber mechanism",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["full", "range_test", "set_home"],
                        "default": "full",
                        "description": "Calibration mode"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="tow_cart",
            description="Enter tow mode: grab IKEA cart and configure for transport",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["attach", "detach", "status"],
                        "default": "attach",
                        "description": "Tow action"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="dock_charger",
            description="Dock with charging station using rear hitch",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["dock", "undock", "status"],
                        "default": "dock",
                        "description": "Docking action"
                    },
                    "verify_charge": {
                        "type": "boolean",
                        "default": True,
                        "description": "Verify charging contact is made"
                    }
                }
            }
        ))

        # =====================================================================
        # FULL SYSTEM SETUP
        # =====================================================================

        self.register(ToolDefinition(
            name="setup_robot",
            description="Run full robot setup sequence",
            parameters={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["scan", "calibrate_arms", "calibrate_gantry", "calibrate_lift", "calibrate_base", "calibrate_hitch", "test_all"]
                        },
                        "default": ["scan", "calibrate_arms", "calibrate_gantry", "calibrate_hitch"],
                        "description": "Which setup steps to run"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="self_test",
            description="Run self-test on all subsystems",
            parameters={
                "type": "object",
                "properties": {
                    "subsystems": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["left_arm", "right_arm", "gantry", "lift", "base", "hitch"]
                        },
                        "default": ["left_arm", "right_arm", "gantry", "hitch"],
                        "description": "Which subsystems to test"
                    },
                    "verbose": {
                        "type": "boolean",
                        "default": False,
                        "description": "Report detailed results"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="save_config",
            description="Save current motor positions and settings as a named configuration",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for this configuration"
                    },
                    "subsystems": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["left_arm", "right_arm", "gantry", "lift", "base", "hitch", "all"]
                        },
                        "default": ["all"],
                        "description": "Which subsystems to save"
                    }
                },
                "required": ["name"]
            }
        ))

        self.register(ToolDefinition(
            name="load_config",
            description="Load a saved configuration",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of configuration to load"
                    }
                },
                "required": ["name"]
            }
        ))

        # =====================================================================
        # RECOGNITION TOOLS (Face + Voice)
        # =====================================================================

        self.register(ToolDefinition(
            name="get_visible_people",
            description="Get list of people currently visible to the camera with their recognition status",
            parameters={
                "type": "object",
                "properties": {
                    "include_unknown": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include unidentified people in results"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="identify_person",
            description="Identify who a specific person is by their position or track ID",
            parameters={
                "type": "object",
                "properties": {
                    "track_id": {
                        "type": "integer",
                        "description": "Track ID from get_visible_people"
                    },
                    "position": {
                        "type": "string",
                        "enum": ["left", "center", "right", "closest"],
                        "description": "Position in frame if track_id not known"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="enroll_person",
            description="Enroll a new person into the recognition database using their current face",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name to assign to this person"
                    },
                    "track_id": {
                        "type": "integer",
                        "description": "Track ID of person to enroll (optional, uses closest if not specified)"
                    },
                    "is_admin": {
                        "type": "boolean",
                        "default": False,
                        "description": "Make this person an admin"
                    }
                },
                "required": ["name"]
            }
        ))

        self.register(ToolDefinition(
            name="get_speaker_direction",
            description="Get the direction (DOA angle) of the current speaker from the microphone array",
            parameters={
                "type": "object",
                "properties": {}
            }
        ))

        self.register(ToolDefinition(
            name="identify_speaker",
            description="Identify who is currently speaking using voice recognition",
            parameters={
                "type": "object",
                "properties": {
                    "use_doa": {
                        "type": "boolean",
                        "default": True,
                        "description": "Also use direction of arrival to disambiguate"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="enroll_voice",
            description="Enroll a person's voice into the speaker recognition database",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of person whose voice to enroll"
                    },
                    "duration_seconds": {
                        "type": "number",
                        "default": 10,
                        "description": "How long to record for enrollment"
                    }
                },
                "required": ["name"]
            }
        ))

        self.register(ToolDefinition(
            name="get_known_people",
            description="Get list of all people in the recognition database",
            parameters={
                "type": "object",
                "properties": {
                    "include_face": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include people with face enrollment"
                    },
                    "include_voice": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include people with voice enrollment"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="forget_person",
            description="Remove a person from the recognition database",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of person to forget"
                    },
                    "forget_face": {
                        "type": "boolean",
                        "default": True,
                        "description": "Forget face encoding"
                    },
                    "forget_voice": {
                        "type": "boolean",
                        "default": True,
                        "description": "Forget voice encoding"
                    }
                },
                "required": ["name"]
            }
        ))

        self.register(ToolDefinition(
            name="who_said_that",
            description="Identify who just spoke (uses recent audio buffer)",
            parameters={
                "type": "object",
                "properties": {
                    "lookback_seconds": {
                        "type": "number",
                        "default": 5,
                        "description": "How far back to check"
                    }
                }
            }
        ))

        self.register(ToolDefinition(
            name="look_at_speaker",
            description="Turn head to look at whoever is currently speaking",
            parameters={
                "type": "object",
                "properties": {}
            }
        ))

        self.register(ToolDefinition(
            name="get_last_seen",
            description="Get when a person was last seen",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of person to check"
                    }
                },
                "required": ["name"]
            }
        ))


def tool(name: str, description: str, **params):
    """Decorator to register a function as a tool.

    Example:
        @tool("wave", "Wave hello", arm={"type": "string", "enum": ["left", "right"]})
        async def wave_handler(arm: str = "right"):
            await robot.wave(arm)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Store tool metadata
        wrapper._tool_definition = ToolDefinition(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": params
            },
            handler=wrapper
        )
        return wrapper
    return decorator


# Global registry instance
_registry = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
