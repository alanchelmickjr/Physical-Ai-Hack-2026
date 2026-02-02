# LED + AEC Integration Plan: Fixing the Feedback Loop Properly

## The Problem We've Been Fighting

Johnny hears herself speak. We've been using **software workarounds**:

```
Current Flow (Broken):
┌─────────────────────────────────────────────────────────┐
│  Johnny speaks → HDMI speaker → sound hits ReSpeaker   │
│       ↓                                                 │
│  ReSpeaker picks up Johnny's voice                      │
│       ↓                                                 │
│  Hume thinks user is speaking → responds → LOOP!        │
└─────────────────────────────────────────────────────────┘

Workarounds Applied:
1. socket.mute() when audio plays     ← Can't interrupt Johnny
2. allow_user_interrupt: False        ← Can't interrupt Johnny
3. 0.3s delay after audio ends        ← Sluggish response
4. PulseAudio WebRTC AEC              ← Partially working
```

**Result:** Johnny works, but feels unnatural. Users can't interrupt. 0.3s dead time after each response.

---

## The Solution: Tell ReSpeaker When We're Speaking

The ReSpeaker has **on-chip AEC** in the XMOS XVF-3000 DSP. It NEEDS to know what audio is being played to cancel it out. Right now it doesn't know.

### Two-Part Fix:

1. **LED Feedback** - Visual state indicator (immediate win)
2. **Proper AEC** - Feed reference audio to ReSpeaker (removes software muting)

---

## Part 1: LED State Machine

### States

| State | LED Effect | Trigger |
|-------|-----------|---------|
| IDLE | Off or dim white | No activity |
| LISTENING | Blue spin | Mic active, waiting for speech |
| THINKING | Purple pulse | User spoke, waiting for Hume response |
| SPEAKING | Green solid | Johnny is outputting audio |
| ERROR | Red flash | Connection error |

### Implementation

```python
# src/led_controller.py
import usb.core
import usb.util
import threading
import time

class ReSpeakerLED:
    """Control ReSpeaker 12-LED ring via USB HID"""

    VENDOR_ID = 0x2886
    PRODUCT_ID = 0x0018

    # LED modes (XMOS firmware commands)
    MODE_OFF = 0
    MODE_LISTEN = 1      # Blue spin
    MODE_SPEAK = 2       # Green
    MODE_THINK = 3       # Purple pulse
    MODE_SPIN = 4        # Custom spin
    MODE_CUSTOM = 5      # Set individual colors

    def __init__(self):
        self.dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
        if not self.dev:
            print("WARNING: ReSpeaker LED not found, LED feedback disabled")
            self.dev = None
        self._current_state = None

    def _write(self, data):
        """Write to USB HID"""
        if not self.dev:
            return
        try:
            self.dev.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, 0, 0, data
            )
        except Exception as e:
            print(f"LED write error: {e}")

    def off(self):
        """Turn off all LEDs"""
        if self._current_state == 'off':
            return
        self._current_state = 'off'
        self._write([0, 0, 0, 0])  # All off

    def listening(self):
        """Blue spinning - Johnny is listening"""
        if self._current_state == 'listening':
            return
        self._current_state = 'listening'
        # Blue color, spin mode
        self._write([1, 0, 0, 255])  # Mode 1, R=0, G=0, B=255

    def thinking(self):
        """Purple pulse - Processing"""
        if self._current_state == 'thinking':
            return
        self._current_state = 'thinking'
        # Purple color, pulse mode
        self._write([3, 128, 0, 255])  # Mode 3, R=128, G=0, B=255

    def speaking(self):
        """Green solid - Johnny is talking"""
        if self._current_state == 'speaking':
            return
        self._current_state = 'speaking'
        # Green color, solid mode
        self._write([2, 0, 255, 0])  # Mode 2, R=0, G=255, B=0

    def error(self):
        """Red flash - Error"""
        self._current_state = 'error'
        self._write([4, 255, 0, 0])  # Mode 4, R=255, G=0, B=0

# Global instance
_led = None

def get_led():
    global _led
    if _led is None:
        _led = ReSpeakerLED()
    return _led
```

### Integration into johnny5.py

```python
# Add to imports
from led_controller import get_led

# In on_message():

elif message.type == "user_message":
    # User spoke - now thinking
    get_led().thinking()
    ...

elif message.type == "audio_output":
    if audio_chunk_count == 0:
        # First audio chunk - Johnny is speaking
        get_led().speaking()
    ...

elif message.type == "assistant_end":
    # Done speaking - back to listening
    get_led().listening()
    ...
```

---

## Part 2: Proper AEC (Echo Cancellation)

### The ReSpeaker's AEC Problem

The XMOS XVF-3000 has **hardware AEC**, but it needs a **reference signal** - the audio being played through the speaker. Without this, it can't know what to cancel.

### Current Setup (Broken AEC)

```
Audio Path:
  Hume → johnny5.py → HDMI speaker

Mic Path:
  ReSpeaker mics → Hume

Problem: ReSpeaker doesn't know what HDMI is playing!
```

### Option A: Use ReSpeaker's 3.5mm Output (Best AEC)

Route Johnny's audio through the ReSpeaker's 3.5mm jack instead of HDMI:

```
Audio Path (Fixed):
  Hume → johnny5.py → ReSpeaker 3.5mm jack → External speaker
                          ↓
                   (AEC reference captured internally)

Mic Path:
  ReSpeaker mics → (AEC applied) → Clean audio → Hume
```

**Pros:** Hardware AEC works perfectly
**Cons:** Need external powered speaker on 3.5mm, not HDMI

**Implementation:**
```bash
# Set ReSpeaker as output device
pactl set-default-sink alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array-00.analog-stereo
```

```python
# In johnny5.py find_audio_devices():
for i, dev in enumerate(devices):
    name = dev['name'].lower()
    if 'respeaker' in name:
        if dev['max_input_channels'] > 0:
            input_device = i
        if dev['max_output_channels'] > 0:  # ReSpeaker has 3.5mm out!
            output_device = i  # Use same device for input AND output
```

### Option B: Software Loopback Reference (Keep HDMI)

Feed a copy of the audio to ReSpeaker while playing through HDMI:

```
Audio Path:
  Hume → johnny5.py → HDMI speaker (for sound)
              ↓
         Also send to ReSpeaker 3.5mm (silent, just for AEC reference)
```

**Implementation with PulseAudio:**
```bash
# Create a combined sink that plays to both HDMI and ReSpeaker
pactl load-module module-combine-sink \
    sink_name=combined \
    slaves=alsa_output.usb-SEEED_ReSpeaker,alsa_output.hdmi \
    adjust_time=0
```

### Option C: PulseAudio WebRTC AEC (Current, Improved)

Keep current setup but optimize the WebRTC AEC module:

```bash
# In start_johnny5.sh - improved settings
pactl load-module module-echo-cancel \
    use_master_format=1 \
    aec_method=webrtc \
    aec_args="analog_gain_control=0 digital_gain_control=1 noise_suppression=1 voice_detection=1 extended_filter=1" \
    source_name=ec_source \
    sink_name=ec_sink \
    source_master=alsa_input.usb-SEEED_ReSpeaker \
    sink_master=alsa_output.hdmi
```

**Key addition:** `extended_filter=1` and `voice_detection=1` for better echo tail handling.

---

## Part 3: Remove Software Workarounds

Once proper AEC is working, we can remove the hacks:

### Before (Current Code)
```python
# johnny5.py line 129-137
if audio_chunk_count == 0:
    # MUTE MIC when audio starts playing (prevent feedback loop)
    audio_playing = True
    if _socket:
        await _socket.mute()

# johnny5.py line 279
"allow_user_interrupt": False,  # DISABLED - prevents feedback loop
```

### After (With Proper AEC)
```python
# johnny5.py - LED feedback only, no muting
if audio_chunk_count == 0:
    get_led().speaking()  # Visual feedback only

# johnny5.py - interrupts enabled!
"allow_user_interrupt": True,  # User can interrupt Johnny naturally
```

---

## Implementation Plan

### Phase 1: LED Feedback (30 min)

1. Create `src/led_controller.py` with USB HID control
2. Add LED state changes to `johnny5.py`:
   - `user_message` → `thinking()`
   - First `audio_output` → `speaking()`
   - `assistant_end` → `listening()`
   - `on_open` → `listening()`
   - `on_error` → `error()`
3. Test: Johnny's ring should change color with state

### Phase 2: Test Current AEC (15 min)

Before changing audio routing, test if current WebRTC AEC is actually working:

```bash
# Check if echo-cancel module is loaded
pactl list modules | grep echo

# Check if ec_source is being used
pactl get-default-source
```

If it shows `ec_source`, AEC is active. If Johnny still hears herself, the AEC isn't getting a good reference signal.

### Phase 3: Choose AEC Strategy (30 min)

**Test each option:**

| Option | Setup | Test |
|--------|-------|------|
| A: ReSpeaker 3.5mm | Connect powered speaker to ReSpeaker 3.5mm, set as output | Does Johnny hear herself? |
| B: Combined sink | Create PulseAudio combined sink | Does Johnny hear herself? |
| C: Better WebRTC | Add extended_filter=1 | Does Johnny hear herself? |

Pick whichever eliminates feedback.

### Phase 4: Enable Interrupts (15 min)

Once AEC works:

1. Change `allow_user_interrupt` to `True`
2. Remove `socket.mute()` / `socket.unmute()` calls
3. Remove 0.3s delay
4. Test: Say "stop" while Johnny is speaking - she should stop!

---

## Hardware Setup for Option A (Best Quality)

If using ReSpeaker's 3.5mm output:

```
Current:
  Jetson HDMI → Monitor speakers

New:
  ReSpeaker 3.5mm → Powered speaker (e.g., USB speaker or 3.5mm amp)
  Jetson HDMI → Monitor (video only, muted)
```

**Powered speaker options:**
- USB-powered PC speaker (~$15)
- 3.5mm to Bluetooth transmitter → BT speaker
- Small class-D amp + speaker

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/led_controller.py` | **NEW** - LED state machine |
| `johnny5.py` | Add LED calls, remove mute workaround |
| `start_johnny5.sh` | Update PulseAudio AEC config |
| `requirements.txt` | Add `pyusb` |

---

## Success Criteria

1. **LEDs reflect state:** Blue listening → Purple thinking → Green speaking → Blue listening
2. **No feedback:** Johnny doesn't hear herself, even at high volume
3. **Interrupts work:** User can say "stop" or "wait" and Johnny responds
4. **No delays:** Immediate transition from speaking to listening
5. **Natural conversation:** Back-and-forth feels fluid, not robotic

---

## Quick Test Commands

```bash
# Install pyusb for LED control
pip install pyusb --break-system-packages

# Test LED control
python3 -c "
from led_controller import get_led
import time
led = get_led()
led.listening(); time.sleep(2)
led.thinking(); time.sleep(2)
led.speaking(); time.sleep(2)
led.off()
"

# Check current audio routing
pactl list sinks short
pactl list sources short
pactl get-default-source
pactl get-default-sink

# Test ReSpeaker 3.5mm output
pactl set-default-sink alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array-00.analog-stereo
speaker-test -t wav -c 2  # Should hear from 3.5mm jack
```

---

## The Big Picture

```
BEFORE (Fighting Feedback):
┌─────────────────────────────────────────────────────────┐
│  Johnny speaks → HDMI                                   │
│       ↓                                                 │
│  Sound hits ReSpeaker (no AEC reference)                │
│       ↓                                                 │
│  Workaround: MUTE MIC (can't interrupt!)                │
│       ↓                                                 │
│  Wait 0.3s after audio ends (sluggish)                  │
│       ↓                                                 │
│  UNMUTE MIC                                             │
└─────────────────────────────────────────────────────────┘

AFTER (Proper AEC):
┌─────────────────────────────────────────────────────────┐
│  Johnny speaks → Speaker (HDMI or 3.5mm)                │
│       ↓              ↓                                  │
│  LED: GREEN    AEC gets reference signal                │
│       ↓              ↓                                  │
│  Sound hits mics     DSP cancels Johnny's voice         │
│       ↓              ↓                                  │
│  Clean audio → Hume (Johnny's voice removed!)           │
│       ↓                                                 │
│  User CAN interrupt naturally!                          │
│       ↓                                                 │
│  LED: BLUE (listening) immediately after speaking       │
└─────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Immediate:** Implement LED controller - visual feedback even before AEC is fixed
2. **Test:** Try ReSpeaker 3.5mm output with a powered speaker
3. **Decide:** Which AEC strategy works best in your environment
4. **Polish:** Remove workarounds, enable interrupts, tune timing
