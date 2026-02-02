#!/usr/bin/env python3
"""Terrain Navigation - Real-World Obstacle Handling

Handles navigation challenges that SLAM can't solve:
- Elevator gaps (accelerate to lift front wheels!)
- Sliding door rails (cross at angle)
- Cords/cables on floor (go around)
- Thresholds and bumps
- Carrying stability

Uses visual detection to predict and time maneuvers.
"I see the gap ahead, timing my acceleration..."

This module provides autonomous handling of terrain challenges
without requiring pre-mapped solutions.

Note: v2 will use hoverboard wheels for better balance.
"""

import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict
from enum import Enum, auto

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class TerrainType(Enum):
    """Types of terrain challenges."""
    ELEVATOR_GAP = auto()      # Gap between elevator and floor (~2 inches)
    DOOR_RAIL = auto()         # Sliding door track on floor
    CORD = auto()              # Power cord, cable on floor
    THRESHOLD = auto()         # Door threshold, small bump
    CARPET_EDGE = auto()       # Carpet to hard floor transition
    RAMP = auto()              # Incline/decline
    GRATE = auto()             # Floor grate, vent
    UNKNOWN_OBSTACLE = auto()


class CarryingState(Enum):
    """What the robot is carrying."""
    EMPTY = auto()
    LIGHT = auto()       # Light items, stable
    FRAGILE = auto()     # Needs careful handling
    LIQUID = auto()      # Don't spill!
    HEAVY = auto()       # Affects balance - need MORE acceleration for wheelie


class CrossingStrategy(Enum):
    """How to cross an obstacle."""
    STRAIGHT_SLOW = auto()     # Slow and steady
    ANGLED = auto()            # Hit at an angle (for rails)
    PERPENDICULAR = auto()     # Cross perpendicular to obstacle
    AVOID = auto()             # Go around it
    STOP = auto()              # Can't cross safely
    MOMENTUM = auto()          # Need momentum to cross
    FRONT_WHEELIE = auto()     # Accelerate to lift front wheels (elevator gaps)


@dataclass
class TerrainObstacle:
    """A detected terrain obstacle."""
    terrain_type: TerrainType
    confidence: float
    x: int  # Pixel position
    y: int
    width: int
    height: int
    distance_mm: float  # Estimated distance from robot
    angle_degrees: float  # Angle from forward
    recommended_strategy: CrossingStrategy
    crossing_speed: float = 0.3  # 0-1, how fast to cross
    needs_announcement: bool = True  # Should robot verbally announce?


@dataclass
class TerrainConfig:
    """Configuration for terrain navigation."""
    enabled: bool = True

    # Detection settings
    cord_detection: bool = True
    gap_detection: bool = True
    rail_detection: bool = True

    # Safety settings
    max_gap_mm: float = 80.0  # Max gap we can wheelie over (~3 inches)
    max_rail_height_mm: float = 15.0  # Max rail we can cross

    # Speed adjustments when carrying
    speed_multiplier_empty: float = 1.0
    speed_multiplier_light: float = 0.8
    speed_multiplier_fragile: float = 0.5
    speed_multiplier_liquid: float = 0.3
    speed_multiplier_heavy: float = 0.6

    # Elevator gap crossing params
    gap_approach_distance_mm: float = 300.0  # Start acceleration this far out
    gap_burst_speed: float = 0.8  # Speed burst to lift front wheels
    gap_burst_duration_ms: float = 200  # How long to accelerate

    # Verbal feedback
    announce_obstacles: bool = True
    announce_crossing: bool = True


class TerrainNavigator:
    """Handles autonomous terrain navigation.

    Works like a person: see obstacle, decide strategy, execute carefully.
    Visual detection lets us TIME maneuvers like the front wheelie for gaps.
    """

    def __init__(self, config: Optional[TerrainConfig] = None):
        self.config = config or TerrainConfig()

        # Callbacks
        self._speak_callback: Optional[Callable[[str], None]] = None
        self._move_callback: Optional[Callable] = None
        self._get_depth_callback: Optional[Callable] = None

        # State
        self._carrying = CarryingState.EMPTY
        self._carrying_description = ""
        self._last_obstacle: Optional[TerrainObstacle] = None
        self._crossing_in_progress = False

        # Wheel load monitoring (for cord detection)
        self._baseline_wheel_loads: List[float] = [0, 0, 0]
        self._sudden_drag_threshold = 1.5  # Multiplier over baseline

        # Gap tracking for timed acceleration
        self._approaching_gap: Optional[TerrainObstacle] = None
        self._gap_first_seen_distance: float = 0.0

        # Announcements made (don't repeat)
        self._announced_obstacles: set = set()

    def set_speak_callback(self, callback: Callable[[str], None]):
        """Set callback for verbal announcements."""
        self._speak_callback = callback

    def set_move_callback(self, callback: Callable):
        """Set callback for movement commands."""
        self._move_callback = callback

    def set_depth_callback(self, callback: Callable):
        """Set callback for depth frame from OAK-D."""
        self._get_depth_callback = callback

    def set_carrying(self, state: CarryingState, description: str = ""):
        """Set what the robot is currently carrying."""
        self._carrying = state
        self._carrying_description = description

        if state == CarryingState.LIQUID:
            self._speak(f"I'm carrying {description}. I'll be extra careful not to spill.")
        elif state == CarryingState.FRAGILE:
            self._speak(f"Carrying something fragile. Taking it slow.")
        elif state == CarryingState.HEAVY:
            self._speak(f"Heavy load. I'll need extra momentum for any gaps.")

    def _speak(self, text: str):
        """Verbal announcement."""
        if self.config.announce_obstacles and self._speak_callback:
            self._speak_callback(text)
        print(f"[Terrain] {text}")

    def _get_speed_multiplier(self) -> float:
        """Get speed multiplier based on what we're carrying."""
        multipliers = {
            CarryingState.EMPTY: self.config.speed_multiplier_empty,
            CarryingState.LIGHT: self.config.speed_multiplier_light,
            CarryingState.FRAGILE: self.config.speed_multiplier_fragile,
            CarryingState.LIQUID: self.config.speed_multiplier_liquid,
            CarryingState.HEAVY: self.config.speed_multiplier_heavy,
        }
        return multipliers.get(self._carrying, 1.0)

    # =========================================================================
    # Visual Detection
    # =========================================================================

    def process_frame(self, rgb_frame: np.ndarray,
                     depth_frame: Optional[np.ndarray] = None) -> List[TerrainObstacle]:
        """Process camera frame for terrain obstacles."""
        if not HAS_OPENCV or not self.config.enabled:
            return []

        obstacles = []
        height, width = rgb_frame.shape[:2]

        # Look at bottom portion of frame (floor area)
        floor_region = rgb_frame[int(height * 0.6):, :]
        floor_depth = depth_frame[int(height * 0.6):, :] if depth_frame is not None else None

        # Detect cords
        if self.config.cord_detection:
            cord = self._detect_cord(floor_region, floor_depth, width, height)
            if cord:
                obstacles.append(cord)

        # Detect gaps (using depth)
        if self.config.gap_detection and floor_depth is not None:
            gap = self._detect_gap(floor_depth, width, height)
            if gap:
                obstacles.append(gap)
                # Track gap for timed acceleration
                self._track_approaching_gap(gap)

        # Detect rails (linear features on floor)
        if self.config.rail_detection:
            rail = self._detect_rail(floor_region, floor_depth, width, height)
            if rail:
                obstacles.append(rail)

        # Announce new obstacles
        for obs in obstacles:
            if obs.needs_announcement:
                self._announce_obstacle(obs)

        return obstacles

    def _track_approaching_gap(self, gap: TerrainObstacle):
        """Track gap distance for timing the acceleration burst."""
        if self._approaching_gap is None:
            self._approaching_gap = gap
            self._gap_first_seen_distance = gap.distance_mm
            self._speak("Elevator gap ahead. Timing my approach...")
        else:
            # Update distance
            self._approaching_gap = gap

            # Check if it's time to accelerate
            if gap.distance_mm <= self.config.gap_approach_distance_mm:
                asyncio.create_task(self._execute_gap_burst())

    async def _execute_gap_burst(self):
        """Execute the acceleration burst to lift front wheels over gap."""
        if self._crossing_in_progress:
            return

        self._crossing_in_progress = True
        gap = self._approaching_gap

        try:
            # Adjust burst based on load
            if self._carrying == CarryingState.HEAVY:
                burst_speed = min(1.0, self.config.gap_burst_speed * 1.3)
                self._speak("Heavy load - extra burst!")
            elif self._carrying == CarryingState.LIQUID:
                # Can't do aggressive wheelie with liquid
                self._speak("Liquid onboard - slow crossing, sorry if we bump.")
                burst_speed = 0.3
            else:
                burst_speed = self.config.gap_burst_speed

            self._speak("Accelerating... now!")

            if self._move_callback:
                # Burst forward - acceleration lifts front wheels
                await self._move_callback("forward", speed=burst_speed, burst=True)
                await asyncio.sleep(self.config.gap_burst_duration_ms / 1000.0)
                # Coast through
                await self._move_callback("forward", speed=0.4)
                await asyncio.sleep(0.3)

            self._speak("Made it across!")
            self._approaching_gap = None

        finally:
            self._crossing_in_progress = False

    def _detect_cord(self, floor_region: np.ndarray,
                    depth: Optional[np.ndarray],
                    frame_width: int, frame_height: int) -> Optional[TerrainObstacle]:
        """Detect cords/cables on floor.

        Cords appear as thin, elongated dark or colored lines.
        """
        gray = cv2.cvtColor(floor_region, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Hough line detection for thin lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=50, maxLineGap=10)

        if lines is None:
            return None

        # Filter for cord-like lines (thin, elongated)
        cord_candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Cords are typically long and thin
            if length > 80:
                cord_candidates.append((x1, y1, x2, y2, length))

        if not cord_candidates:
            return None

        # Take longest cord candidate
        cord = max(cord_candidates, key=lambda c: c[4])
        x1, y1, x2, y2, length = cord

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2 + int(frame_height * 0.6)

        distance = self._estimate_distance_from_y(center_y, frame_height)
        angle = self._pixel_to_angle(center_x, frame_width)

        return TerrainObstacle(
            terrain_type=TerrainType.CORD,
            confidence=0.7,
            x=center_x, y=center_y,
            width=int(length), height=10,
            distance_mm=distance,
            angle_degrees=angle,
            recommended_strategy=CrossingStrategy.AVOID,
            crossing_speed=0.0,  # Don't cross, go around
            needs_announcement=True
        )

    def _detect_gap(self, depth_frame: np.ndarray,
                   frame_width: int, frame_height: int) -> Optional[TerrainObstacle]:
        """Detect gaps using depth discontinuity.

        Elevator gaps show as sudden depth increase (floor drops away).
        Visual detection lets us TIME the acceleration burst.
        """
        if depth_frame is None:
            return None

        # Look at horizontal slice of depth
        floor_depth = depth_frame[int(depth_frame.shape[0] * 0.7):, :]

        # Find sudden depth changes (gap = floor drops away)
        depth_gradient = np.gradient(floor_depth.astype(float), axis=0)

        # Large positive gradient = floor dropping away
        gap_mask = depth_gradient > 50  # Threshold in mm

        if not np.any(gap_mask):
            return None

        # Find gap region
        gap_cols = np.where(np.any(gap_mask, axis=0))[0]
        if len(gap_cols) < 20:  # Too small
            return None

        center_x = int(np.mean(gap_cols))
        center_y = int(frame_height * 0.85)
        gap_width = gap_cols[-1] - gap_cols[0]

        distance = self._estimate_distance_from_y(center_y, frame_height)
        angle = self._pixel_to_angle(center_x, frame_width)

        # Determine if crossable with front wheelie technique
        if gap_width > self.config.max_gap_mm:
            strategy = CrossingStrategy.STOP
            speed = 0.0
        else:
            # Front wheelie: accelerate to lift front wheels over gap
            strategy = CrossingStrategy.FRONT_WHEELIE
            speed = self.config.gap_burst_speed

        return TerrainObstacle(
            terrain_type=TerrainType.ELEVATOR_GAP,
            confidence=0.8,
            x=center_x, y=center_y,
            width=gap_width, height=20,
            distance_mm=distance,
            angle_degrees=angle,
            recommended_strategy=strategy,
            crossing_speed=speed,
            needs_announcement=True
        )

    def _detect_rail(self, floor_region: np.ndarray,
                    depth: Optional[np.ndarray],
                    frame_width: int, frame_height: int) -> Optional[TerrainObstacle]:
        """Detect door rails / tracks on floor.

        Rails appear as straight lines across the floor, often metallic/reflective.
        """
        gray = cv2.cvtColor(floor_region, cv2.COLOR_BGR2GRAY)

        # Look for horizontal lines (rails usually cross path)
        edges = cv2.Canny(gray, 30, 100)

        # Detect near-horizontal lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=100, maxLineGap=20)

        if lines is None:
            return None

        # Filter for horizontal-ish lines (rails across path)
        rail_candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Rails are roughly horizontal (within 30 degrees)
            if angle < 30 or angle > 150:
                if length > 100:
                    rail_candidates.append((x1, y1, x2, y2, length, angle))

        if not rail_candidates:
            return None

        # Take longest rail
        rail = max(rail_candidates, key=lambda r: r[4])
        x1, y1, x2, y2, length, line_angle = rail

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2 + int(frame_height * 0.6)

        distance = self._estimate_distance_from_y(center_y, frame_height)
        angle = self._pixel_to_angle(center_x, frame_width)

        # Rails should be crossed at an angle, not straight
        return TerrainObstacle(
            terrain_type=TerrainType.DOOR_RAIL,
            confidence=0.6,
            x=center_x, y=center_y,
            width=int(length), height=15,
            distance_mm=distance,
            angle_degrees=angle,
            recommended_strategy=CrossingStrategy.ANGLED,  # Hit at angle!
            crossing_speed=0.4 * self._get_speed_multiplier(),
            needs_announcement=True
        )

    def _estimate_distance_from_y(self, y: int, frame_height: int) -> float:
        """Rough distance estimate from vertical position in frame."""
        normalized = (frame_height - y) / frame_height
        return 300 + normalized * 1700

    def _pixel_to_angle(self, x: int, frame_width: int) -> float:
        """Convert pixel x to angle from center."""
        normalized = (x - frame_width / 2) / (frame_width / 2)
        return normalized * 35  # Assume 70 degree FOV

    # =========================================================================
    # Crossing Execution
    # =========================================================================

    async def cross_obstacle(self, obstacle: TerrainObstacle) -> bool:
        """Execute crossing strategy for an obstacle."""
        if self._crossing_in_progress:
            return False

        self._crossing_in_progress = True
        self._last_obstacle = obstacle

        try:
            strategy = obstacle.recommended_strategy

            if strategy == CrossingStrategy.STOP:
                self._speak("I can't cross this safely. Need to find another way.")
                return False

            elif strategy == CrossingStrategy.AVOID:
                return await self._execute_avoid(obstacle)

            elif strategy == CrossingStrategy.ANGLED:
                return await self._execute_angled_crossing(obstacle)

            elif strategy == CrossingStrategy.STRAIGHT_SLOW:
                return await self._execute_slow_crossing(obstacle)

            elif strategy == CrossingStrategy.PERPENDICULAR:
                return await self._execute_perpendicular_crossing(obstacle)

            elif strategy == CrossingStrategy.MOMENTUM:
                return await self._execute_momentum_crossing(obstacle)

            elif strategy == CrossingStrategy.FRONT_WHEELIE:
                return await self._execute_front_wheelie_crossing(obstacle)

        finally:
            self._crossing_in_progress = False

        return True

    async def _execute_avoid(self, obstacle: TerrainObstacle) -> bool:
        """Go around the obstacle."""
        if obstacle.terrain_type == TerrainType.CORD:
            self._speak("There's a cord on the floor. Going around it.")

        # Determine which way to go around
        if obstacle.angle_degrees > 0:
            direction = "left"
        else:
            direction = "right"

        if self._move_callback:
            await self._move_callback("strafe", direction, distance=0.3)
            await asyncio.sleep(0.5)
            await self._move_callback("forward", distance=0.5)
            await asyncio.sleep(0.5)
            opposite = "right" if direction == "left" else "left"
            await self._move_callback("strafe", opposite, distance=0.3)

        return True

    async def _execute_angled_crossing(self, obstacle: TerrainObstacle) -> bool:
        """Cross at an angle (for rails)."""
        self._speak("Door rail ahead. Crossing at an angle.")

        if self._carrying == CarryingState.LIQUID:
            self._speak("Taking it extra slow with the liquid.")

        speed = obstacle.crossing_speed

        if self._move_callback:
            # Turn slightly
            await self._move_callback("rotate", degrees=15)
            await asyncio.sleep(0.3)

            # Cross slowly
            await self._move_callback("forward", distance=0.3, speed=speed)
            await asyncio.sleep(0.5)

            # Straighten out
            await self._move_callback("rotate", degrees=-15)

        self._speak("Made it across.")
        return True

    async def _execute_slow_crossing(self, obstacle: TerrainObstacle) -> bool:
        """Cross slowly and carefully."""
        self._speak("Obstacle ahead. Going slow.")

        speed = obstacle.crossing_speed

        if self._move_callback:
            await self._move_callback("forward", distance=0.2, speed=speed)
            await asyncio.sleep(1.0)
            await self._move_callback("forward", distance=0.2, speed=speed)

        self._speak("Clear.")
        return True

    async def _execute_perpendicular_crossing(self, obstacle: TerrainObstacle) -> bool:
        """Cross perpendicular to obstacle (for thresholds)."""
        self._speak("Threshold ahead. Crossing straight on.")

        if self._move_callback:
            if abs(obstacle.angle_degrees) > 5:
                await self._move_callback("rotate", degrees=-obstacle.angle_degrees)

            await self._move_callback("forward", distance=0.3, speed=obstacle.crossing_speed)

        return True

    async def _execute_momentum_crossing(self, obstacle: TerrainObstacle) -> bool:
        """Cross with momentum (for small bumps)."""
        self._speak("Small bump. Using a bit of momentum.")

        if self._carrying in (CarryingState.LIQUID, CarryingState.FRAGILE):
            self._speak("Actually, carrying something delicate. Taking it slow instead.")
            return await self._execute_slow_crossing(obstacle)

        if self._move_callback:
            await self._move_callback("forward", distance=0.4, speed=0.6)

        return True

    async def _execute_front_wheelie_crossing(self, obstacle: TerrainObstacle) -> bool:
        """Cross gap by accelerating to lift front wheels.

        The key insight: sudden acceleration shifts weight backward,
        lifting the front wheels. Time it so front wheels are UP
        when hitting the gap edge.

        Visual detection lets us see the gap coming and time it right.
        """
        self._speak("Elevator gap! Timing my acceleration...")

        # Adjust for load
        if self._carrying == CarryingState.HEAVY:
            burst_speed = min(1.0, self.config.gap_burst_speed * 1.3)
            self._speak("Heavy load - extra burst needed!")
        elif self._carrying == CarryingState.LIQUID:
            self._speak("Can't wheelie with liquid. Slow and bumpy, sorry.")
            if self._move_callback:
                await self._move_callback("forward", distance=0.4, speed=0.25)
            return True
        else:
            burst_speed = self.config.gap_burst_speed

        if self._move_callback:
            # Acceleration burst - front wheels lift
            self._speak("Now!")
            await self._move_callback("forward", speed=burst_speed, burst=True)
            await asyncio.sleep(self.config.gap_burst_duration_ms / 1000.0)

            # Coast through while front wheels clear
            await self._move_callback("forward", speed=0.5)
            await asyncio.sleep(0.4)

        self._speak("Clear!")
        return True

    # =========================================================================
    # Wheel Drag Detection (Cord caught)
    # =========================================================================

    def update_wheel_loads(self, loads: List[float]):
        """Update wheel load readings for drag detection."""
        if not self._baseline_wheel_loads[0]:
            self._baseline_wheel_loads = loads.copy()
            return

        for i, (current, baseline) in enumerate(zip(loads, self._baseline_wheel_loads)):
            if baseline > 0 and current > baseline * self._sudden_drag_threshold:
                self._on_wheel_drag_detected(i)
                return

        alpha = 0.1
        for i in range(len(loads)):
            self._baseline_wheel_loads[i] = (
                alpha * loads[i] + (1 - alpha) * self._baseline_wheel_loads[i]
            )

    def _on_wheel_drag_detected(self, wheel_index: int):
        """Handle detected wheel drag (cord caught?)."""
        wheel_names = ["front", "back-left", "back-right"]
        wheel = wheel_names[wheel_index] if wheel_index < len(wheel_names) else f"wheel {wheel_index}"

        self._speak(f"Something caught on my {wheel} wheel! Stopping!")

        if self._move_callback:
            asyncio.create_task(self._emergency_stop_and_reverse())

    async def _emergency_stop_and_reverse(self):
        """Stop and reverse slightly to free caught cord."""
        if self._move_callback:
            await self._move_callback("stop")
            await asyncio.sleep(0.3)
            await self._move_callback("backward", distance=0.1, speed=0.2)

        self._speak("Backed up a bit. Let me look for what I caught.")

    # =========================================================================
    # Announcements
    # =========================================================================

    def _announce_obstacle(self, obstacle: TerrainObstacle):
        """Verbally announce detected obstacle."""
        key = f"{obstacle.terrain_type}_{int(obstacle.distance_mm)}"
        if key in self._announced_obstacles:
            return

        self._announced_obstacles.add(key)

        if len(self._announced_obstacles) > 20:
            self._announced_obstacles.clear()

        messages = {
            TerrainType.ELEVATOR_GAP: "I see an elevator gap ahead. I'll time my acceleration.",
            TerrainType.DOOR_RAIL: "There's a door rail on the floor.",
            TerrainType.CORD: "Watch out, there's a cord on the floor.",
            TerrainType.THRESHOLD: "Door threshold coming up.",
            TerrainType.CARPET_EDGE: "Carpet edge ahead.",
            TerrainType.RAMP: "There's a ramp here.",
            TerrainType.GRATE: "Floor grate ahead.",
        }

        msg = messages.get(obstacle.terrain_type, "Obstacle detected.")
        self._speak(msg)


# Singleton
_navigator: Optional[TerrainNavigator] = None


def get_terrain_navigator() -> TerrainNavigator:
    """Get the singleton terrain navigator."""
    global _navigator
    if _navigator is None:
        _navigator = TerrainNavigator()
    return _navigator
