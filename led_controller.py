#!/usr/bin/env python3
"""ReSpeaker USB Mic Array LED Controller

Controls the 12 RGB LEDs on the ReSpeaker USB Mic Array v2.0.
Uses the pixel_ring library or direct USB HID if pixel_ring unavailable.

States:
- LISTENING: Blue spin (DOA mode)
- THINKING: Purple pulse
- SPEAKING: Green solid
- ERROR: Red flash
"""

import time
import threading

# Try pixel_ring first (Seeed's official library)
try:
    from pixel_ring import pixel_ring
    PIXEL_RING_AVAILABLE = True
    print("LED Controller: Using pixel_ring library")
except ImportError:
    PIXEL_RING_AVAILABLE = False
    print("LED Controller: pixel_ring not found, trying direct USB")

# Fallback to direct USB HID
if not PIXEL_RING_AVAILABLE:
    try:
        import usb.core
        import usb.util
        USB_AVAILABLE = True
    except ImportError:
        USB_AVAILABLE = False
        print("WARNING: Neither pixel_ring nor pyusb installed")
        print("  pip install pixel_ring --break-system-packages")
        print("  OR pip install pyusb --break-system-packages")
else:
    USB_AVAILABLE = False  # Don't need raw USB if pixel_ring works


class ReSpeakerLED:
    """Control ReSpeaker 4-Mic Array LEDs"""

    VENDOR_ID = 0x2886   # Seeed Studio
    PRODUCT_ID = 0x0018  # ReSpeaker USB Mic Array

    def __init__(self):
        self.dev = None
        self._current_state = None
        self._lock = threading.Lock()
        self._use_pixel_ring = PIXEL_RING_AVAILABLE

        if self._use_pixel_ring:
            try:
                pixel_ring.set_brightness(20)
                print("LED Controller: pixel_ring initialized")
            except Exception as e:
                print(f"LED Controller: pixel_ring init failed: {e}")
                self._use_pixel_ring = False

        if not self._use_pixel_ring and USB_AVAILABLE:
            try:
                self.dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
                if self.dev is None:
                    print("LED Controller: ReSpeaker not found on USB")
                else:
                    if self.dev.is_kernel_driver_active(0):
                        try:
                            self.dev.detach_kernel_driver(0)
                        except usb.core.USBError:
                            pass
                    print("LED Controller: Direct USB initialized")
            except Exception as e:
                print(f"LED Controller: USB init error: {e}")
                self.dev = None

    def _usb_set_color(self, r: int, g: int, b: int):
        """Set all LEDs via direct USB (fallback)"""
        if not self.dev:
            return
        try:
            # ReSpeaker USB protocol: send RGB to all 12 LEDs
            # Command 0x00 with color data
            self.dev.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, 0, 0,
                [0, r, g, b] * 12,  # 12 LEDs, each gets RGB
                timeout=1000
            )
        except Exception as e:
            print(f"LED USB error: {e}")

    def off(self):
        """Turn off all LEDs"""
        with self._lock:
            if self._current_state == 'off':
                return
            self._current_state = 'off'
            if self._use_pixel_ring:
                pixel_ring.off()
            else:
                self._usb_set_color(0, 0, 0)
            print("LED: OFF")

    def listening(self):
        """Blue spin - listening for speech (DOA mode)"""
        with self._lock:
            if self._current_state == 'listening':
                return
            self._current_state = 'listening'
            if self._use_pixel_ring:
                pixel_ring.wakeup()  # Blue spinning DOA effect
            else:
                self._usb_set_color(0, 0, 255)
            print("LED: LISTENING (blue)")

    def thinking(self):
        """Purple pulse - processing"""
        with self._lock:
            if self._current_state == 'thinking':
                return
            self._current_state = 'thinking'
            if self._use_pixel_ring:
                pixel_ring.think()  # Purple breathing
            else:
                self._usb_set_color(128, 0, 255)
            print("LED: THINKING (purple)")

    def speaking(self):
        """Green solid - Johnny is talking"""
        with self._lock:
            if self._current_state == 'speaking':
                return
            self._current_state = 'speaking'
            if self._use_pixel_ring:
                pixel_ring.speak()  # Green
            else:
                self._usb_set_color(0, 255, 0)
            print("LED: SPEAKING (green)")

    def error(self):
        """Red - error state"""
        with self._lock:
            self._current_state = 'error'
            if self._use_pixel_ring:
                pixel_ring.set_color(rgb=0xFF0000)  # Red
            else:
                self._usb_set_color(255, 0, 0)
            print("LED: ERROR (red)")

    @property
    def state(self):
        return self._current_state


# Singleton
_led_instance = None
_led_lock = threading.Lock()


def get_led() -> ReSpeakerLED:
    """Get the singleton LED controller"""
    global _led_instance
    with _led_lock:
        if _led_instance is None:
            _led_instance = ReSpeakerLED()
        return _led_instance


if __name__ == "__main__":
    print("Testing ReSpeaker LED Controller")
    print("=" * 40)

    led = get_led()

    print("\n1. Listening (blue)")
    led.listening()
    time.sleep(3)

    print("\n2. Thinking (purple)")
    led.thinking()
    time.sleep(3)

    print("\n3. Speaking (green)")
    led.speaking()
    time.sleep(3)

    print("\n4. Error (red)")
    led.error()
    time.sleep(3)

    print("\n5. Off")
    led.off()

    print("\nDone!")
