#!/usr/bin/env python3
"""Verbal Calibration System

Interactive calibration where the robot and human work together
to identify and configure motors through conversation.

Flow:
1. Robot scans motors, reports what it finds
2. Human provides hints ("those are probably the base motors")
3. Robot tests motors one by one
4. Robot uses camera to verify movement
5. Human confirms or corrects

Example conversation:
    Robot: "I found 10 motors on ACM0 and 8 on ACM1.
           Let me check what's connected..."
    Robot: "Motors 7, 8, 9 on ACM0 don't seem to respond.
           Could those be the base wheels?"
    Human: "The AlohaMini docs say 8, 9, 10. Try those."
    Robot: "Testing motor 8... I see something moving.
           Is that a wheel?"
    Human: "Yes, that's the front wheel"
    Robot: "Got it. Motor 8 is front wheel. Testing 9..."
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
from enum import Enum

from config.hardware import HardwareConfig, get_hardware_config
from config.motors import MotorInterface, MotorDiscovery, get_motor_interface


class CalibrationState(Enum):
    """States in the calibration flow."""
    IDLE = "idle"
    SCANNING = "scanning"
    AWAITING_INPUT = "awaiting_input"
    TESTING_MOTOR = "testing_motor"
    VERIFYING_VISUAL = "verifying_visual"
    CONFIRMING = "confirming"
    COMPLETE = "complete"


# MotorDiscovery imported from config.motors (single source of truth)


@dataclass
class CalibrationSession:
    """State of an ongoing calibration session."""
    state: CalibrationState = CalibrationState.IDLE
    discovered_motors: Dict[str, MotorDiscovery] = field(default_factory=dict)
    current_motor: Optional[str] = None
    pending_question: Optional[str] = None
    human_responses: List[str] = field(default_factory=list)
    subsystem_being_calibrated: Optional[str] = None


class VerbalCalibration:
    """Interactive calibration assistant.

    Works with Hume EVI for voice interaction and
    OAK-D camera for visual verification.

    Message Flow:
        Spine → VerbalCalibration → johnny5.py (Hume EVI) → Human
        Human → johnny5.py → VerbalCalibration → Spine

    The calibration sends messages via a callback, and receives
    human responses via process_human_response().
    """

    def __init__(self):
        self.session = CalibrationSession()

        # Shared config and motor interface (single source of truth)
        self.config = get_hardware_config()
        self.motors = get_motor_interface()

        # Callback to send message to johnny5.py for speaking
        self._speak_callback: Optional[Callable[[str], None]] = None

        # Callback to get camera frame for motion detection
        self._camera_callback: Optional[Callable[[], Any]] = None

        # Callback to send tool result back to johnny5.py
        self._result_callback: Optional[Callable[[dict], None]] = None

        # Message queue for async communication
        self._pending_message: Optional[str] = None
        self._awaiting_response: bool = False

        # Expected layout derived from shared config
        # ACM0: Left arm (1-6), lift (10), wheels (7-9)
        # ACM1: Right arm (1-6), gantry (7-8)
        self.expected_layout = {
            "ACM0": {
                "left_arm": list(self.config.LEFT_ARM_IDS),
                "base": list(self.config.WHEEL_IDS),
                "lift": [self.config.LIFT_ID],
            },
            "ACM1": {
                "right_arm": list(self.config.RIGHT_ARM_IDS),
                "gantry": list(self.config.GANTRY_IDS),
            }
        }

        # Alternative layouts from different docs
        # AlohaMini sometimes uses 8,9,10 for base
        self.alternative_layouts = {
            "aloha_mini_base": [8, 9, 10],
            "aloha_mini_lift": [7],  # If they swapped lift and first wheel
        }

        # Verbal descriptions for each subsystem
        self.subsystem_descriptions = {
            "left_arm": "the left SO101 arm, motors 1 through 6",
            "right_arm": "the right SO101 arm, motors 1 through 6",
            "base": "the three mecanum wheels for the mobile base",
            "gantry": "the camera pan and tilt servos",
            "lift": "the 30 centimeter lift mechanism",
        }

    def set_speak_callback(self, callback: Callable[[str], None]):
        """Set callback to speak to human (via Hume EVI)."""
        self._speak_callback = callback

    def set_camera_callback(self, callback: Callable[[], Any]):
        """Set callback to get camera frame for motion detection."""
        self._camera_callback = callback

    def set_result_callback(self, callback: Callable[[dict], None]):
        """Set callback to send tool results back to johnny5.py."""
        self._result_callback = callback

    def _speak(self, text: str):
        """Speak to the human via johnny5.py/Hume EVI."""
        self._pending_message = text
        self._awaiting_response = True
        if self._speak_callback:
            self._speak_callback(text)
        print(f"[CALIBRATION → JOHNNY5]: {text}")

    def _send_result(self, success: bool, message: str, data: dict = None):
        """Send result back to the tool execution engine."""
        result = {
            "success": success,
            "message": message,
            "data": data or {},
            "awaiting_response": self._awaiting_response
        }
        if self._result_callback:
            self._result_callback(result)
        print(f"[CALIBRATION RESULT]: {result}")

    def is_awaiting_response(self) -> bool:
        """Check if calibration is waiting for human response."""
        return self._awaiting_response

    def get_pending_message(self) -> Optional[str]:
        """Get the message waiting to be spoken to human."""
        return self._pending_message

    async def start_calibration(self, subsystem: Optional[str] = None) -> str:
        """Start a calibration session.

        Returns initial message to speak to human.
        """
        self.session = CalibrationSession(
            state=CalibrationState.SCANNING,
            subsystem_being_calibrated=subsystem
        )

        # Scan motors
        scan_results = await self._scan_all_motors()

        # Build response
        acm0_count = len([m for m in scan_results.values() if m.bus == "ACM0" and m.responds])
        acm1_count = len([m for m in scan_results.values() if m.bus == "ACM1" and m.responds])

        self.session.discovered_motors = scan_results
        self.session.state = CalibrationState.AWAITING_INPUT

        msg = f"I found {acm0_count} motors on ACM0 and {acm1_count} on ACM1. "

        # Check for discrepancies
        expected_acm0 = 10  # 6 arm + 3 wheels + 1 lift
        expected_acm1 = 8   # 6 arm + 2 gantry

        if acm0_count != expected_acm0:
            msg += f"I expected {expected_acm0} on ACM0. "
        if acm1_count != expected_acm1:
            msg += f"I expected {expected_acm1} on ACM1. "

        # Check specific IDs
        missing = self._find_missing_motors(scan_results)
        if missing:
            msg += f"Motors {missing} don't seem to respond. "

        if subsystem == "base":
            msg += "Should I test the base motors? The docs say IDs 7, 8, 9 but some AlohaMini versions use 8, 9, 10."

        return msg

    async def _scan_all_motors(self) -> Dict[str, MotorDiscovery]:
        """Scan both buses for motors.

        Delegates to config.motors.MotorInterface (single source of truth).
        """
        results = {}

        # Scan ACM0 (IDs 1-10)
        for motor_id in range(1, 11):
            key = f"ACM0:{motor_id}"
            responds = await self.motors.ping(self.config.LEFT_PORT, motor_id)
            results[key] = MotorDiscovery(
                bus="ACM0",
                motor_id=motor_id,
                responds=responds
            )

        # Scan ACM1 (IDs 1-8)
        for motor_id in range(1, 9):
            key = f"ACM1:{motor_id}"
            responds = await self.motors.ping(self.config.RIGHT_PORT, motor_id)
            results[key] = MotorDiscovery(
                bus="ACM1",
                motor_id=motor_id,
                responds=responds
            )

        return results

    async def _ping_motor(self, port: str, motor_id: int) -> bool:
        """Ping a motor to see if it responds.

        Delegates to config.motors.MotorInterface (single source of truth).
        """
        return await self.motors.ping(port, motor_id)

    def _find_missing_motors(self, scan_results: Dict[str, MotorDiscovery]) -> List[str]:
        """Find motors that should exist but don't respond."""
        missing = []

        # Check ACM0 expected motors
        for subsystem, ids in self.expected_layout["ACM0"].items():
            for motor_id in ids:
                key = f"ACM0:{motor_id}"
                if key in scan_results and not scan_results[key].responds:
                    missing.append(f"ACM0:{motor_id} ({subsystem})")

        # Check ACM1 expected motors
        for subsystem, ids in self.expected_layout["ACM1"].items():
            for motor_id in ids:
                key = f"ACM1:{motor_id}"
                if key in scan_results and not scan_results[key].responds:
                    missing.append(f"ACM1:{motor_id} ({subsystem})")

        return missing

    async def process_human_response(self, response: str) -> str:
        """Process human's response and continue calibration.

        Args:
            response: What the human said

        Returns:
            Next message to speak
        """
        self.session.human_responses.append(response)
        response_lower = response.lower()

        # Check for alternative motor IDs
        if "8" in response and "9" in response and "10" in response:
            # Human suggests alternative base motor IDs
            self._speak("Got it, trying motors 8, 9, 10 for the base.")
            return await self._test_motor_group("ACM0", [8, 9, 10], "base")

        # Check for confirmation
        if any(word in response_lower for word in ["yes", "correct", "right", "that's it"]):
            if self.session.current_motor:
                # Confirm current motor identification
                motor = self.session.discovered_motors.get(self.session.current_motor)
                if motor:
                    motor.human_confirmed = True
                    return await self._test_next_motor()

        # Check for correction
        if any(word in response_lower for word in ["no", "wrong", "not", "different"]):
            if self.session.current_motor:
                self._speak("Okay, what is this motor then?")
                self.session.state = CalibrationState.AWAITING_INPUT
                return "What should I identify this motor as?"

        # Check for motor type identification
        motor_types = ["wheel", "arm", "gripper", "gantry", "lift", "pan", "tilt"]
        for mtype in motor_types:
            if mtype in response_lower:
                if self.session.current_motor:
                    motor = self.session.discovered_motors.get(self.session.current_motor)
                    if motor:
                        motor.identified_as = mtype
                        motor.human_confirmed = True
                        return await self._test_next_motor()

        # Check for "try" or "test" commands
        if "try" in response_lower or "test" in response_lower:
            # Extract motor ID if mentioned
            import re
            ids = re.findall(r'\b(\d+)\b', response)
            if ids:
                motor_id = int(ids[0])
                # Determine bus from context
                bus = "ACM0"  # Default
                if "right" in response_lower or "acm1" in response_lower:
                    bus = "ACM1"
                return await self._test_single_motor(bus, motor_id)

        # Default: ask for clarification
        return "I didn't understand. Could you tell me which motors to test, or confirm what you see?"

    async def _test_motor_group(self, bus: str, motor_ids: List[int], group_name: str) -> str:
        """Test a group of motors (e.g., base wheels)."""
        results = []

        for motor_id in motor_ids:
            key = f"{bus}:{motor_id}"
            self.session.current_motor = key

            # Test motor
            moved = await self._wiggle_motor(bus, motor_id)

            motor = self.session.discovered_motors.get(key)
            if motor:
                motor.tested = True
                motor.movement_detected = moved
                motor.identified_as = f"{group_name}_{motor_id}"

            results.append((motor_id, moved))

        # Build response
        moved_ids = [mid for mid, moved in results if moved]
        failed_ids = [mid for mid, moved in results if not moved]

        if moved_ids and not failed_ids:
            return f"All motors {motor_ids} responded. I detected movement on all of them. Can you confirm these are the {group_name} motors?"
        elif moved_ids:
            return f"Motors {moved_ids} moved, but {failed_ids} didn't respond. Did you see the {group_name} move?"
        else:
            return f"None of motors {motor_ids} seemed to move. Are they connected? Should I try different IDs?"

    async def _test_single_motor(self, bus: str, motor_id: int) -> str:
        """Test a single motor and ask human to identify it."""
        key = f"{bus}:{motor_id}"
        self.session.current_motor = key
        self.session.state = CalibrationState.TESTING_MOTOR

        # Wiggle the motor
        moved = await self._wiggle_motor(bus, motor_id)

        # Try to detect with camera
        visual_detected = await self._detect_motion_with_camera()

        motor = self.session.discovered_motors.get(key)
        if motor:
            motor.tested = True
            motor.movement_detected = moved or visual_detected

        self.session.state = CalibrationState.CONFIRMING

        if visual_detected:
            return f"Testing motor {motor_id} on {bus}. I can see something moving! What part is that?"
        elif moved:
            return f"Testing motor {motor_id} on {bus}. The motor responded. Did you see anything move?"
        else:
            return f"Motor {motor_id} on {bus} doesn't seem to respond. Is it connected?"

    async def _test_next_motor(self) -> str:
        """Move to testing the next unidentified motor."""
        # Find next untested motor
        for key, motor in self.session.discovered_motors.items():
            if motor.responds and not motor.human_confirmed:
                return await self._test_single_motor(motor.bus, motor.motor_id)

        # All done
        self.session.state = CalibrationState.COMPLETE
        return self._generate_summary()

    async def _wiggle_motor(self, bus: str, motor_id: int) -> bool:
        """Move a motor slightly and back to test it.

        Delegates to config.motors.MotorInterface (single source of truth).
        """
        port = self.config.LEFT_PORT if bus == "ACM0" else self.config.RIGHT_PORT
        return await self.motors.wiggle(port, motor_id)

    async def _detect_motion_with_camera(self) -> bool:
        """Use camera to detect if something moved."""
        if not self._camera_callback:
            return False

        try:
            # Get frame before
            frame_before = self._camera_callback()
            await asyncio.sleep(0.5)
            # Get frame after
            frame_after = self._camera_callback()

            # Simple motion detection (would use OpenCV diff in real impl)
            # For now, return False (camera integration TBD)
            return False
        except:
            return False

    def _generate_summary(self) -> str:
        """Generate summary of calibration results."""
        identified = []
        unidentified = []

        for key, motor in self.session.discovered_motors.items():
            if motor.human_confirmed and motor.identified_as:
                identified.append(f"{key}: {motor.identified_as}")
            elif motor.responds:
                unidentified.append(key)

        summary = "Calibration complete! "
        if identified:
            summary += f"Identified {len(identified)} motors. "
        if unidentified:
            summary += f"{len(unidentified)} motors still need identification. "

        summary += "Should I save this configuration?"
        return summary

    def get_current_layout(self) -> Dict:
        """Get the current motor layout based on calibration."""
        layout = {"ACM0": {}, "ACM1": {}}

        for key, motor in self.session.discovered_motors.items():
            if motor.human_confirmed and motor.identified_as:
                bus = motor.bus
                subsystem = motor.identified_as.split("_")[0]  # e.g., "wheel" from "wheel_8"

                if subsystem not in layout[bus]:
                    layout[bus][subsystem] = []
                layout[bus][subsystem].append(motor.motor_id)

        return layout


# Singleton
_calibration: Optional[VerbalCalibration] = None


def get_verbal_calibration() -> VerbalCalibration:
    """Get the verbal calibration instance."""
    global _calibration
    if _calibration is None:
        _calibration = VerbalCalibration()
    return _calibration
