# Tool Execution Engine for Hume EVI Integration
#
# This module provides the tool execution engine that processes
# Hume EVI tool calls and routes them to the robot adapter.
#
# Key components:
# - ToolExecutionEngine: Processes Hume EVI tool calls
# - ToolRegistry: Defines available tools
# - RealTimeCommandQueue: Bridges cloud latency to real-time robot
# - RecognitionHandler: Bridges recognition queries via IPC

from .engine import ToolExecutionEngine, ToolCall, ToolResult, ActionGraph
from .registry import ToolRegistry, tool
from .realtime import (
    RealTimeCommandQueue,
    RealTimeExecutor,
    TimedCommand,
    CommandPriority,
    get_command_queue,
)
from .recognition_handler import (
    RecognitionHandler,
    RecognitionConfig,
    create_recognition_handler,
)

__all__ = [
    # Engine
    "ToolExecutionEngine",
    "ToolCall",
    "ToolResult",
    "ActionGraph",
    # Registry
    "ToolRegistry",
    "tool",
    # Real-time
    "RealTimeCommandQueue",
    "RealTimeExecutor",
    "TimedCommand",
    "CommandPriority",
    "get_command_queue",
    # Recognition
    "RecognitionHandler",
    "RecognitionConfig",
    "create_recognition_handler",
]
