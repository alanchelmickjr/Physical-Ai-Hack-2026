"""Real-Time Command Queue for Robot Control

This module bridges the gap between cloud-based AI (Hume EVI)
and real-time robot execution.

Problem:
- Hume EVI has variable latency (100-500ms round trip)
- Robot needs consistent timing (10-30ms control loop)
- Commands may arrive while robot is mid-action

Solution:
- Local command queue with timestamps
- Priority system for emergency commands
- Action interpolation for smooth motion
- Timing correction for network jitter
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional
import threading


class CommandPriority(IntEnum):
    """Priority levels for commands."""
    EMERGENCY = 0    # Stop, safety - executes IMMEDIATELY
    HIGH = 1         # User explicit request
    NORMAL = 2       # AI-initiated action
    LOW = 3          # Background/idle behavior
    DEFERRED = 4     # Execute when idle


@dataclass
class TimedCommand:
    """A command with timing information."""
    command: Dict[str, Any]
    priority: CommandPriority = CommandPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    deadline: Optional[float] = None  # Execute before this time
    interruptible: bool = True
    callback: Optional[Callable] = None

    @property
    def age_ms(self) -> float:
        """How old is this command in milliseconds."""
        return (time.time() - self.timestamp) * 1000

    @property
    def is_expired(self) -> bool:
        """Has the deadline passed?"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline


class RealTimeCommandQueue:
    """Thread-safe command queue with timing and priority.

    This queue sits between Hume EVI (async, cloud latency) and
    the robot control loop (sync, real-time).

    Features:
    - Priority-based ordering
    - Timestamp tracking
    - Deadline expiration
    - Emergency command bypass
    - Smooth action blending
    """

    def __init__(self, max_queue_size: int = 100):
        self.queue: deque[TimedCommand] = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()
        self._emergency_flag = threading.Event()
        self._current_command: Optional[TimedCommand] = None
        self._stats = {
            "commands_received": 0,
            "commands_executed": 0,
            "commands_expired": 0,
            "emergency_stops": 0,
            "avg_latency_ms": 0.0,
        }

    def push(self, command: TimedCommand) -> bool:
        """Add a command to the queue.

        Emergency commands are handled immediately.

        Args:
            command: The command to queue

        Returns:
            True if queued, False if dropped
        """
        with self._lock:
            self._stats["commands_received"] += 1

            # Emergency commands bypass the queue
            if command.priority == CommandPriority.EMERGENCY:
                self._emergency_flag.set()
                self._stats["emergency_stops"] += 1
                # Clear queue of interruptible commands
                self.queue = deque(
                    [c for c in self.queue if not c.interruptible],
                    maxlen=self.queue.maxlen
                )
                # Insert at front
                self.queue.appendleft(command)
                return True

            # Check for expired deadline
            if command.is_expired:
                self._stats["commands_expired"] += 1
                return False

            # Insert based on priority
            # Lower priority number = higher priority
            inserted = False
            for i, existing in enumerate(self.queue):
                if command.priority < existing.priority:
                    self.queue.insert(i, command)
                    inserted = True
                    break

            if not inserted:
                self.queue.append(command)

            return True

    def pop(self) -> Optional[TimedCommand]:
        """Get the next command to execute.

        Returns:
            Next command, or None if queue is empty
        """
        with self._lock:
            # Remove expired commands
            while self.queue:
                cmd = self.queue[0]
                if cmd.is_expired:
                    self.queue.popleft()
                    self._stats["commands_expired"] += 1
                    continue
                break

            if not self.queue:
                return None

            cmd = self.queue.popleft()
            self._current_command = cmd
            self._stats["commands_executed"] += 1

            # Update latency stats
            latency = cmd.age_ms
            alpha = 0.1  # Exponential moving average
            self._stats["avg_latency_ms"] = (
                alpha * latency +
                (1 - alpha) * self._stats["avg_latency_ms"]
            )

            return cmd

    def peek(self) -> Optional[TimedCommand]:
        """Look at the next command without removing it."""
        with self._lock:
            if not self.queue:
                return None
            return self.queue[0]

    def clear(self) -> int:
        """Clear all commands from the queue.

        Returns:
            Number of commands cleared
        """
        with self._lock:
            count = len(self.queue)
            self.queue.clear()
            return count

    def is_emergency(self) -> bool:
        """Check if emergency flag is set."""
        return self._emergency_flag.is_set()

    def clear_emergency(self) -> None:
        """Clear the emergency flag."""
        self._emergency_flag.clear()

    @property
    def size(self) -> int:
        """Number of commands in queue."""
        return len(self.queue)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return self._stats.copy()


class RealTimeExecutor:
    """Real-time execution loop that processes commands at consistent rate.

    This runs on a separate thread with high priority, ensuring
    consistent timing regardless of what else is happening.
    """

    def __init__(
        self,
        command_queue: RealTimeCommandQueue,
        execute_fn: Callable,
        loop_rate_hz: int = 30
    ):
        self.queue = command_queue
        self.execute_fn = execute_fn
        self.loop_rate_hz = loop_rate_hz
        self.loop_period_s = 1.0 / loop_rate_hz

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_loop_time = 0.0
        self._loop_stats = {
            "actual_rate_hz": 0.0,
            "overruns": 0,
            "total_loops": 0,
        }

    def start(self) -> None:
        """Start the real-time execution loop."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the execution loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        """Main execution loop - runs at consistent rate."""
        while self._running:
            loop_start = time.perf_counter()

            # Check for emergency
            if self.queue.is_emergency():
                # Execute emergency immediately
                cmd = self.queue.pop()
                if cmd:
                    try:
                        self.execute_fn(cmd.command)
                    except Exception as e:
                        print(f"Emergency execution error: {e}")
                self.queue.clear_emergency()
                continue

            # Normal command processing
            cmd = self.queue.pop()
            if cmd:
                try:
                    self.execute_fn(cmd.command)
                    if cmd.callback:
                        cmd.callback(True, None)
                except Exception as e:
                    print(f"Command execution error: {e}")
                    if cmd.callback:
                        cmd.callback(False, e)

            # Timing
            elapsed = time.perf_counter() - loop_start
            sleep_time = self.loop_period_s - elapsed

            self._loop_stats["total_loops"] += 1

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self._loop_stats["overruns"] += 1

            # Update rate stats
            now = time.perf_counter()
            if self._last_loop_time > 0:
                actual_period = now - self._last_loop_time
                actual_rate = 1.0 / actual_period if actual_period > 0 else 0
                alpha = 0.1
                self._loop_stats["actual_rate_hz"] = (
                    alpha * actual_rate +
                    (1 - alpha) * self._loop_stats["actual_rate_hz"]
                )
            self._last_loop_time = now

    @property
    def stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self._loop_stats.copy()


class ActionInterpolator:
    """Smooth interpolation between robot positions.

    When a new command arrives, instead of jumping to the target,
    this interpolates smoothly over time for natural motion.
    """

    def __init__(self, smoothing_factor: float = 0.1):
        self.smoothing = smoothing_factor
        self.current_positions: Dict[str, List[float]] = {}
        self.target_positions: Dict[str, List[float]] = {}

    def set_target(self, subsystem: str, positions: List[float]) -> None:
        """Set new target positions."""
        self.target_positions[subsystem] = positions
        if subsystem not in self.current_positions:
            self.current_positions[subsystem] = positions.copy()

    def step(self) -> Dict[str, List[float]]:
        """Compute one step of interpolation.

        Returns:
            Current interpolated positions for each subsystem
        """
        for subsystem in self.target_positions:
            if subsystem not in self.current_positions:
                self.current_positions[subsystem] = self.target_positions[subsystem].copy()
                continue

            current = self.current_positions[subsystem]
            target = self.target_positions[subsystem]

            # Exponential smoothing
            for i in range(min(len(current), len(target))):
                current[i] += self.smoothing * (target[i] - current[i])

        return self.current_positions.copy()

    def is_settled(self, tolerance: float = 0.1) -> bool:
        """Check if all positions have reached their targets."""
        for subsystem in self.target_positions:
            if subsystem not in self.current_positions:
                return False

            current = self.current_positions[subsystem]
            target = self.target_positions[subsystem]

            for c, t in zip(current, target):
                if abs(c - t) > tolerance:
                    return False

        return True


# Singleton queue instance
_command_queue: Optional[RealTimeCommandQueue] = None


def get_command_queue() -> RealTimeCommandQueue:
    """Get the global command queue."""
    global _command_queue
    if _command_queue is None:
        _command_queue = RealTimeCommandQueue()
    return _command_queue
