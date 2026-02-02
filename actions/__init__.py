"""Unified Action Primitives

Provides a hardware-agnostic interface for robot actions.
Works across Johnny Five, Booster K1, Unitree G1, and future platforms.

Usage:
    from robot_factory import create_robot
    from actions import UnifiedActions, Target, Hand, GestureType

    robot = create_robot()
    await robot.adapter.connect()

    actions = UnifiedActions(robot)

    # Gestures
    await actions.wave()
    await actions.wave(hand=Hand.LEFT)
    await actions.gesture(GestureType.THUMBS_UP)

    # Head movement
    await actions.look_at(Target.from_angle(45))
    await actions.nod()

    # Locomotion
    await actions.walk_to(Target.from_position(x=1.0, y=0.0))
    await actions.turn(angle=90)

    # Manipulation
    await actions.point_at(Target.from_detection(person))
    await actions.reach(Target.from_position(0.3, 0.1, 0.2))
    await actions.grab()
"""

from .base import (
    ActionStatus,
    ActionResult,
    ActionCapability,
    ActionPrimitives,
    Target,
    Hand,
    GestureType,
    EmotionType,
)
from .unified import UnifiedActions

__all__ = [
    # Core classes
    'UnifiedActions',
    'ActionPrimitives',
    'ActionResult',
    'ActionStatus',
    'ActionCapability',

    # Types
    'Target',
    'Hand',
    'GestureType',
    'EmotionType',
]
