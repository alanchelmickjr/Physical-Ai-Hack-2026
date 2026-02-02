"""Motor Interface - Consolidated motor operations

Single implementation for all motor operations.
Uses Solo-CLI for Feetech servo communication.

Usage:
    from config.motors import MotorInterface

    motors = MotorInterface()
    await motors.ping("/dev/ttyACM0", 1)
    await motors.scan("/dev/ttyACM0", range(1, 11))
    await motors.wiggle("/dev/ttyACM0", 7)
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from config.hardware import get_hardware_config, HardwareConfig


@dataclass
class ServoHealth:
    """Health status of a servo."""
    motor_id: int
    bus: str
    responding: bool = True
    temperature: float = 0.0
    load: float = 0.0
    voltage: float = 0.0
    error_code: int = 0
    last_check: float = 0.0


@dataclass
class MotorDiscovery:
    """Discovery result for a motor."""
    bus: str
    motor_id: int
    responds: bool = False
    identified_as: Optional[str] = None
    tested: bool = False
    movement_detected: bool = False
    human_confirmed: bool = False
    notes: str = ""


class MotorInterface:
    """Unified interface for all motor operations.

    This is the ONLY place motor commands should be issued.
    All other modules should use this interface.
    """

    def __init__(self, config: Optional[HardwareConfig] = None):
        self.config = config or get_hardware_config()
        self._health_cache: Dict[str, ServoHealth] = {}

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def ping(self, port: str, motor_id: int, timeout: float = 2.0) -> bool:
        """Ping a motor to check if it responds.

        Args:
            port: Serial port (e.g., /dev/ttyACM0)
            motor_id: Motor ID to ping
            timeout: Timeout in seconds

        Returns:
            True if motor responds, False otherwise
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "solo", "robo",
                "--port", port,
                "--ids", str(motor_id),
                "--ping",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(process.communicate(), timeout=timeout)
            return process.returncode == 0
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    async def scan(
        self,
        port: str,
        id_range: range,
        timeout: float = 2.0
    ) -> Dict[int, bool]:
        """Scan a range of motor IDs on a port.

        Args:
            port: Serial port
            id_range: Range of IDs to scan
            timeout: Timeout per motor

        Returns:
            Dict mapping motor_id to responds (bool)
        """
        results = {}
        for motor_id in id_range:
            results[motor_id] = await self.ping(port, motor_id, timeout)
        return results

    async def scan_all(self) -> Dict[str, Dict[int, bool]]:
        """Scan all expected motors on both buses.

        Returns:
            Dict mapping bus name to {motor_id: responds}
        """
        results = {}

        # Scan ACM0 (IDs 1-10)
        results["ACM0"] = await self.scan(
            self.config.LEFT_PORT,
            range(1, 11)
        )

        # Scan ACM1 (IDs 1-8)
        results["ACM1"] = await self.scan(
            self.config.RIGHT_PORT,
            range(1, 9)
        )

        return results

    async def wiggle(
        self,
        port: str,
        motor_id: int,
        timeout: float = 5.0
    ) -> bool:
        """Wiggle a motor slightly to test it.

        Args:
            port: Serial port
            motor_id: Motor ID
            timeout: Timeout in seconds

        Returns:
            True if wiggle command succeeded
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "solo", "robo",
                "--port", port,
                "--ids", str(motor_id),
                "--wiggle",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(process.communicate(), timeout=timeout)
            return process.returncode == 0
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    async def move(
        self,
        port: str,
        motor_id: int,
        position: float,
        speed: float = 0.5,
        timeout: float = 5.0
    ) -> bool:
        """Move a motor to a position.

        Args:
            port: Serial port
            motor_id: Motor ID
            position: Target position (degrees or mm depending on motor)
            speed: Movement speed (0.0 - 1.0)
            timeout: Timeout in seconds

        Returns:
            True if move command succeeded
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "solo", "robo",
                "--port", port,
                "--ids", str(motor_id),
                "--position", str(position),
                "--speed", str(speed),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(process.communicate(), timeout=timeout)
            return process.returncode == 0
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    async def read(
        self,
        port: str,
        motor_id: int,
        timeout: float = 2.0
    ) -> Optional[float]:
        """Read current position of a motor.

        Args:
            port: Serial port
            motor_id: Motor ID
            timeout: Timeout in seconds

        Returns:
            Current position, or None if read failed
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "solo", "robo",
                "--port", port,
                "--ids", str(motor_id),
                "--read",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            if process.returncode == 0:
                # Parse position from stdout
                try:
                    return float(stdout.decode().strip())
                except ValueError:
                    return None
            return None
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    # =========================================================================
    # Health Checking
    # =========================================================================

    async def check_health(
        self,
        port: str,
        motor_id: int
    ) -> ServoHealth:
        """Get health status of a servo.

        Args:
            port: Serial port
            motor_id: Motor ID

        Returns:
            ServoHealth with current status
        """
        import time

        bus = "ACM0" if "ACM0" in port else "ACM1"
        key = f"{bus}:{motor_id}"

        responding = await self.ping(port, motor_id)

        health = ServoHealth(
            motor_id=motor_id,
            bus=bus,
            responding=responding,
            last_check=time.time()
        )

        self._health_cache[key] = health
        return health

    async def check_all_health(self) -> Dict[str, ServoHealth]:
        """Check health of all expected motors.

        Returns:
            Dict mapping "bus:id" to ServoHealth
        """
        results = {}

        # Check ACM0
        for motor_id in range(1, 11):
            key = f"ACM0:{motor_id}"
            results[key] = await self.check_health(
                self.config.LEFT_PORT,
                motor_id
            )

        # Check ACM1
        for motor_id in range(1, 9):
            key = f"ACM1:{motor_id}"
            results[key] = await self.check_health(
                self.config.RIGHT_PORT,
                motor_id
            )

        return results

    # =========================================================================
    # Subsystem Operations
    # =========================================================================

    async def move_gantry(
        self,
        pan: float,
        tilt: float,
        speed: float = 0.5
    ) -> bool:
        """Move the camera gantry to a pan/tilt position.

        Args:
            pan: Pan angle in degrees (-90 to 90)
            tilt: Tilt angle in degrees (-45 to 45)
            speed: Movement speed (0.0 - 1.0)

        Returns:
            True if both moves succeeded
        """
        # Clamp to limits
        pan = max(self.config.PAN_LIMITS[0],
                  min(self.config.PAN_LIMITS[1], pan))
        tilt = max(self.config.TILT_LIMITS[0],
                   min(self.config.TILT_LIMITS[1], tilt))

        pan_ok = await self.move(
            self.config.RIGHT_PORT,
            self.config.GANTRY_IDS[0],  # Pan = 7
            pan,
            speed
        )
        tilt_ok = await self.move(
            self.config.RIGHT_PORT,
            self.config.GANTRY_IDS[1],  # Tilt = 8
            tilt,
            speed
        )

        return pan_ok and tilt_ok

    async def center_gantry(self, speed: float = 0.3) -> bool:
        """Center the camera gantry."""
        return await self.move_gantry(0, 0, speed)

    # =========================================================================
    # Startup Calibration (Movie-style "Number 5 is ALIVE!")
    # =========================================================================

    async def startup_calibration(
        self,
        speak_callback: Optional[callable] = None,
        skip_base: bool = False
    ) -> Dict[str, bool]:
        """Run full startup diagnostic sequence.

        Like Johnny Five in the movie - wiggles all servos on boot.

        Args:
            speak_callback: Optional function to announce progress
            skip_base: Skip wheel motors if robot shouldn't move

        Returns:
            Dict mapping subsystem to success status
        """
        import time

        results = {
            "scan": False,
            "gantry": False,
            "left_arm": False,
            "right_arm": False,
            "lift": False,
            "base": False,
        }

        def speak(text: str):
            if speak_callback:
                speak_callback(text)
            print(f"[STARTUP] {text}")

        speak("Good morning! Running startup diagnostics.")

        # 1. Scan all motors
        speak("Scanning all motor banks...")
        health = await self.check_all_health()
        responding = sum(1 for h in health.values() if h.responding)
        total = len(health)
        results["scan"] = responding > 0
        speak(f"Found {responding} of {total} motors responding.")

        # 2. Center gantry
        speak("Centering camera.")
        results["gantry"] = await self.center_gantry(0.3)

        # Small delay between subsystems
        await asyncio.sleep(0.3)

        # 3. Test gantry range
        await self.move_gantry(-45, 0, 0.5)
        await asyncio.sleep(0.2)
        await self.move_gantry(45, 0, 0.5)
        await asyncio.sleep(0.2)
        await self.move_gantry(0, -30, 0.5)
        await asyncio.sleep(0.2)
        await self.move_gantry(0, 30, 0.5)
        await asyncio.sleep(0.2)
        await self.center_gantry(0.3)
        speak("Camera ready.")

        # 4. Wiggle left arm
        speak("Testing left arm.")
        left_arm_ok = True
        for motor_id in self.config.LEFT_ARM_IDS:
            ok = await self.wiggle(self.config.LEFT_PORT, motor_id)
            left_arm_ok = left_arm_ok and ok
            await asyncio.sleep(0.1)
        results["left_arm"] = left_arm_ok

        # 5. Wiggle right arm
        speak("Testing right arm.")
        right_arm_ok = True
        for motor_id in self.config.RIGHT_ARM_IDS:
            ok = await self.wiggle(self.config.RIGHT_PORT, motor_id)
            right_arm_ok = right_arm_ok and ok
            await asyncio.sleep(0.1)
        results["right_arm"] = right_arm_ok

        # 6. Test lift
        speak("Testing lift.")
        lift_ok = await self.move(
            self.config.LEFT_PORT,
            self.config.LIFT_ID,
            200, 0.3
        )
        await asyncio.sleep(0.3)
        await self.move(self.config.LEFT_PORT, self.config.LIFT_ID, 50, 0.3)
        await asyncio.sleep(0.3)
        await self.move(self.config.LEFT_PORT, self.config.LIFT_ID, 150, 0.3)
        results["lift"] = lift_ok
        speak("Lift ready.")

        # 7. Test base (if not skipped)
        if not skip_base:
            speak("Testing base motors.")
            base_ok = True
            for motor_id in self.config.WHEEL_IDS:
                ok = await self.wiggle(self.config.LEFT_PORT, motor_id)
                base_ok = base_ok and ok
            results["base"] = base_ok
            speak("Base motors ready.")
        else:
            results["base"] = True  # Skipped counts as success
            speak("Base motors skipped.")

        # Final status
        all_ok = all(results.values())
        if all_ok:
            speak("All systems nominal. Ready for interaction.")
        else:
            failed = [k for k, v in results.items() if not v]
            speak(f"Warning: Issues detected with {', '.join(failed)}.")

        return results


# =============================================================================
# Singleton
# =============================================================================
_interface: MotorInterface = None


def get_motor_interface() -> MotorInterface:
    """Get the singleton motor interface."""
    global _interface
    if _interface is None:
        _interface = MotorInterface()
    return _interface
