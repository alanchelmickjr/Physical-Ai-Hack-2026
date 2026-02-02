"""Unitree G1 Microphone Array

4-mic array built into the Unitree G1 robot head.
Accessed via unitree_sdk2 or ROS2.
"""

import threading
import time
from typing import Optional

from .base import MicrophoneArray, MicrophoneType, DOAReading


class UnitreeMic(MicrophoneArray):
    """Unitree G1 built-in 4-mic array.

    The G1 has a built-in microphone array in its head.
    DOA and VAD are accessed through unitree_sdk2 or ROS2.

    Falls back to simulation if SDK not available.
    """

    def __init__(self, mic_type: MicrophoneType = MicrophoneType.UNITREE_4MIC):
        super().__init__()
        self._mic_type = mic_type
        self._num_mics = 4
        self._sample_rate = 16000

        # SDK access
        self._sdk_available = False
        self._ros_available = False
        self._simulation_mode = False

        # Try unitree_sdk2 first
        try:
            import unitree_sdk2py
            self._sdk_available = True
            self._sdk = unitree_sdk2py
            print("[UnitreeMic] unitree_sdk2 available")
        except ImportError:
            print("[UnitreeMic] unitree_sdk2 not available")

        # Try ROS2
        if not self._sdk_available:
            try:
                import rclpy
                self._ros_available = True
                print("[UnitreeMic] ROS2 available for audio")
            except ImportError:
                print("[UnitreeMic] ROS2 not available")

        # Fall back to simulation
        if not self._sdk_available and not self._ros_available:
            self._simulation_mode = True
            print("[UnitreeMic] Running in simulation mode")

        # State
        self._current_doa: float = 0.0
        self._current_vad: bool = False
        self._thread: Optional[threading.Thread] = None

        # ROS2 subscriber (if available)
        self._ros_node = None
        self._audio_sub = None

    @property
    def mic_type(self) -> MicrophoneType:
        return self._mic_type

    @property
    def num_mics(self) -> int:
        return self._num_mics

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self, poll_rate_hz: int = 30) -> bool:
        """Start DOA tracking.

        Args:
            poll_rate_hz: Update rate for DOA polling

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        self._running = True

        if self._ros_available:
            self._start_ros()
        else:
            # Poll-based or simulation
            self._thread = threading.Thread(
                target=self._poll_loop,
                args=(poll_rate_hz,),
                daemon=True
            )
            self._thread.start()

        mode = "ROS2" if self._ros_available else (
            "SDK" if self._sdk_available else "simulation"
        )
        print(f"[UnitreeMic] Started ({mode} mode) at {poll_rate_hz}Hz")
        return True

    def stop(self):
        """Stop DOA tracking."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._ros_node:
            self._ros_node.destroy_node()
            self._ros_node = None

        print("[UnitreeMic] Stopped")

    def read_doa_raw(self) -> Optional[int]:
        """Read raw DOA from Unitree microphone array.

        Returns:
            DOA angle in degrees (0-360), or None
        """
        if self._simulation_mode:
            # Return last simulated value
            return int(self._current_doa)

        if self._sdk_available:
            try:
                # Access via unitree_sdk2
                # The actual API depends on unitree_sdk2 version
                # This is a placeholder for the actual implementation
                return int(self._current_doa)
            except Exception as e:
                print(f"[UnitreeMic] SDK DOA read error: {e}")
                return None

        # ROS2 mode - value updated by subscriber
        return int(self._current_doa)

    def read_vad(self) -> bool:
        """Read Voice Activity Detection status.

        Returns:
            True if voice is detected
        """
        if self._simulation_mode:
            return self._current_vad

        if self._sdk_available:
            try:
                # Access via unitree_sdk2
                return self._current_vad
            except Exception:
                return False

        return self._current_vad

    def _start_ros(self):
        """Initialize ROS2 subscriber for audio data."""
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float32MultiArray

            if not rclpy.ok():
                rclpy.init()

            class AudioSubscriber(Node):
                def __init__(self, parent):
                    super().__init__('unitree_mic_subscriber')
                    self.parent = parent
                    # Subscribe to audio DOA topic (topic name may vary)
                    self.subscription = self.create_subscription(
                        Float32MultiArray,
                        '/unitree/audio/doa',
                        self.doa_callback,
                        10
                    )

                def doa_callback(self, msg):
                    if msg.data:
                        self.parent._current_doa = msg.data[0]
                        self.parent._current_vad = len(msg.data) > 1 and msg.data[1] > 0.5
                        self.parent._update_doa(
                            self.parent._current_doa,
                            self.parent._current_vad
                        )
                        self.parent._fire_callbacks(
                            self.parent._current_doa,
                            self.parent._current_vad
                        )

            self._ros_node = AudioSubscriber(self)

            # Spin in background
            self._thread = threading.Thread(
                target=lambda: rclpy.spin(self._ros_node),
                daemon=True
            )
            self._thread.start()

        except Exception as e:
            print(f"[UnitreeMic] ROS2 init failed: {e}")
            self._ros_available = False
            self._simulation_mode = True

    def _poll_loop(self, poll_rate_hz: int):
        """Polling loop for SDK or simulation mode."""
        interval = 1.0 / poll_rate_hz

        while self._running:
            start = time.time()

            if self._simulation_mode:
                # Simulate occasional voice from random directions
                import random
                if random.random() < 0.1:  # 10% chance of voice
                    self._current_doa = random.uniform(0, 360)
                    self._current_vad = True
                else:
                    self._current_vad = False
            else:
                # Read from SDK
                doa = self.read_doa_raw()
                vad = self.read_vad()
                if doa is not None:
                    self._current_doa = doa
                self._current_vad = vad

            # Update smoothed values
            self._update_doa(self._current_doa, self._current_vad)

            # Fire callbacks if speaking
            if self._current_vad:
                self._fire_callbacks(self._current_doa, self._current_vad)

            # Sleep for remainder of interval
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)


# Quick test
if __name__ == "__main__":
    print("Testing Unitree Microphone Array")
    print("=" * 50)

    mic = UnitreeMic()

    def on_doa(doa: float, speaking: bool):
        if speaking:
            print(f"  DOA: {doa:.1f}° (speaking)")

    mic.on_direction_change(on_doa)
    mic.start(poll_rate_hz=30)

    try:
        while True:
            doa, speaking, conf = mic.get_doa()
            status = "SPEAKING" if speaking else "silent"
            print(f"DOA: {doa:5.1f}° | {status}     ", end='\r')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        mic.stop()
        print("Test complete!")
