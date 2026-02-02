#!/usr/bin/env python3
"""ReSpeaker USB Mic Array LED Controller

Controls the 12 RGB LEDs on the ReSpeaker via USB HID.
Provides visual feedback for Johnny Five's conversation state.

States:
- LISTENING: Default DOA mode (let firmware handle it)
- THINKING: Purple pulse (processing user input)
- SPEAKING: Green solid (Johnny is talking)
- ERROR: Red flash

The ReSpeaker's default firmware shows DOA (direction of arrival) with the LEDs.
We override this ONLY when Johnny is speaking, then return to default mode.
"""

import time
import threading

try:
    import usb.core
    import usb.util
    USB_AVAILABLE = True
except ImportError:
    USB_AVAILABLE = False
    print("WARNING: pyusb not installed. Run: pip install pyusb --break-system-packages")


class ReSpeakerLED:
    """Control ReSpeaker 4-Mic Array LEDs via USB HID"""

    VENDOR_ID = 0x2886   # Seeed Studio
    PRODUCT_ID = 0x0018  # ReSpeaker 4-Mic Array

    # XMOS XVF-3000 Tuning registers
    REGISTER_LED_MODE = 0x00
    REGISTER_LED_BRIGHTNESS = 0x01
    REGISTER_LED_COLOR = 0x02

    # LED Modes
    MODE_OFF = 0
    MODE_ON = 1           # Solid color
    MODE_BREATHE = 2      # Breathing/pulse
    MODE_SPIN = 3         # Spinning
    MODE_DOA = 4          # Direction of arrival (default firmware)

    def __init__(self):
        self.dev = None
        self._current_state = None
        self._lock = threading.Lock()

        if not USB_AVAILABLE:
            print("LED Controller: USB not available, running in dummy mode")
            return

        try:
            self.dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
            if self.dev is None:
                print("LED Controller: ReSpeaker not found on USB")
            else:
                # Detach kernel driver if attached
                if self.dev.is_kernel_driver_active(0):
                    try:
                        self.dev.detach_kernel_driver(0)
                    except usb.core.USBError:
                        pass
                print("LED Controller: ReSpeaker found and ready")
        except Exception as e:
            print(f"LED Controller: USB init error: {e}")
            self.dev = None

    def _write_register(self, register: int, value: int) -> bool:
        """Write a value to an XMOS register"""
        if not self.dev:
            return False

        try:
            self.dev.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,  # bRequest
                value,  # wValue
                register,  # wIndex
                [],  # data
                timeout=1000
            )
            return True
        except Exception as e:
            print(f"LED write error: {e}")
            return False

    def _set_color(self, r: int, g: int, b: int) -> bool:
        """Set LED color (RGB, 0-255 each)"""
        if not self.dev:
            return False

        try:
            # Pack RGB into the color register format
            # Different firmware versions may use different formats
            self.dev.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,
                0,  # wValue
                self.REGISTER_LED_COLOR,
                [r, g, b, 0],  # RGB + padding
                timeout=1000
            )
            return True
        except Exception as e:
            print(f"LED color error: {e}")
            return False

    def _set_all_leds(self, r: int, g: int, b: int, brightness: int = 31):
        """Set all 12 LEDs to the same color"""
        if not self.dev:
            return

        try:
            # For pixel_ring compatible firmware, use this approach
            # Each LED needs 4 bytes: LED_index, R, G, B
            for i in range(12):
                self.dev.ctrl_transfer(
                    usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                    0,
                    i,  # LED index in wValue
                    0x1C,  # Custom LED control register
                    [r, g, b, brightness],
                    timeout=1000
                )
        except Exception as e:
            # Try alternative method
            self._write_register(self.REGISTER_LED_MODE, self.MODE_ON)
            self._set_color(r, g, b)

    def off(self):
        """Turn off all LEDs"""
        with self._lock:
            if self._current_state == 'off':
                return
            self._current_state = 'off'
            self._write_register(self.REGISTER_LED_MODE, self.MODE_OFF)
            print("LED: OFF")

    def listening(self):
        """Return to default DOA mode (firmware handles LED direction)"""
        with self._lock:
            if self._current_state == 'listening':
                return
            self._current_state = 'listening'
            # Mode 4 = DOA mode, let firmware show sound direction
            self._write_register(self.REGISTER_LED_MODE, self.MODE_DOA)
            print("LED: LISTENING (DOA mode)")

    def thinking(self):
        """Purple pulse - processing user input"""
        with self._lock:
            if self._current_state == 'thinking':
                return
            self._current_state = 'thinking'
            self._write_register(self.REGISTER_LED_MODE, self.MODE_BREATHE)
            self._set_color(128, 0, 255)  # Purple
            print("LED: THINKING (purple pulse)")

    def speaking(self):
        """Green solid - Johnny is outputting audio"""
        with self._lock:
            if self._current_state == 'speaking':
                return
            self._current_state = 'speaking'
            self._write_register(self.REGISTER_LED_MODE, self.MODE_ON)
            self._set_color(0, 255, 0)  # Green
            self._set_all_leds(0, 255, 0)  # Ensure all LEDs are green
            print("LED: SPEAKING (green solid)")

    def error(self):
        """Red flash - error state"""
        with self._lock:
            self._current_state = 'error'
            self._write_register(self.REGISTER_LED_MODE, self.MODE_BREATHE)
            self._set_color(255, 0, 0)  # Red
            print("LED: ERROR (red flash)")

    @property
    def state(self):
        """Get current LED state"""
        return self._current_state


# Singleton instance
_led_instance = None
_led_lock = threading.Lock()


def get_led() -> ReSpeakerLED:
    """Get the singleton LED controller instance"""
    global _led_instance
    with _led_lock:
        if _led_instance is None:
            _led_instance = ReSpeakerLED()
        return _led_instance


# Quick test
if __name__ == "__main__":
    print("Testing ReSpeaker LED Controller")
    print("=" * 40)

    led = get_led()

    print("\n1. Listening mode (DOA - default)")
    led.listening()
    time.sleep(3)

    print("\n2. Thinking mode (purple pulse)")
    led.thinking()
    time.sleep(3)

    print("\n3. Speaking mode (green solid)")
    led.speaking()
    time.sleep(3)

    print("\n4. Error mode (red flash)")
    led.error()
    time.sleep(3)

    print("\n5. Back to listening")
    led.listening()
    time.sleep(2)

    print("\n6. Off")
    led.off()

    print("\nTest complete!")
