# Bugs and Improvements

21 issues found during code review. Organized by severity and category.

---

## Critical: Disconnected Components

### 1. `johnny5.py` doesn't integrate the Spine
**File:** `johnny5.py`
**Type:** Missing Integration

The main Hume EVI entry point (`johnny5.py`) doesn't import or use:
- `motion_coordinator` (the spine)
- `johnny5_body` (body abstraction)
- `visual_safety` (fire detection)
- `terrain_navigation` (obstacle handling)

The "mind" and "body" are completely disconnected.

```python
# johnny5.py has NO imports for:
# from motion_coordinator import get_motion_coordinator
# from visual_safety import get_visual_safety
```

**Impact:** All autonomic features (head tracking, gestures, fire detection) don't run.

---

### 2. `johnny5.py` doesn't handle tool_call messages
**File:** `johnny5.py:112-190`
**Type:** Missing Handler

The `on_message()` handler processes `user_message`, `assistant_message`, `audio_output`, but has NO handler for `tool_call` or `tool_use` message types.

```python
# Missing:
elif message.type == "tool_call":
    # Parse and execute tool
    result = await tool_engine.execute_tool(...)
```

**Impact:** Hume can't actually control the robot - tool calls are ignored.

---

### 3. `ToolExecutionEngine` never instantiated
**File:** `tools/engine.py`, `johnny5.py`
**Type:** Dead Code

`ToolExecutionEngine` is defined but never instantiated anywhere. No code creates an engine or connects it to Hume.

```python
# tools/__init__.py exports it:
from .engine import ToolExecutionEngine

# But nobody imports and uses it:
# engine = ToolExecutionEngine(robot_adapter)  # <- nowhere
```

**Impact:** 50+ tools defined in registry.py can never execute.

---

### 4. `remember_person()` never called
**File:** `motion_coordinator.py:410`
**Type:** Dead Code / Missing Integration

The `_known_people` dict is used for pointing at people:
```python
for name in names_mentioned:
    if name in self._known_people:  # Always empty!
        await self._point_at_person(name)
```

But `remember_person(name, doa)` is **never called** from face recognition or anywhere else.

**Impact:** Polite pointing feature never activates.

---

### 5. `process_speech_text()` never called
**File:** `motion_coordinator.py:331-358`, `johnny5.py`
**Type:** Missing Integration

The spine's `process_speech_text()` method generates gestures based on what the robot is saying:
```python
async def process_speech_text(self, text: str, names_mentioned: List[str] = None):
    """Process text being spoken, generating appropriate gestures."""
```

But `johnny5.py` never calls this when Hume speaks.

**Impact:** Robot doesn't gesture while talking.

---

### 6. `set_speaking()` never called
**File:** `motion_coordinator.py`, `johnny5.py`
**Type:** Missing Integration

The spine has `set_speaking(is_speaking: bool)` to know when the robot is talking (affects head tracking responsiveness).

```python
# johnny5.py should call:
coordinator.set_speaking(True)   # when audio starts
coordinator.set_speaking(False)  # when audio ends
```

But it doesn't.

**Impact:** Head tracking doesn't adapt during speech.

---

## High: Logic Errors

### 7. Duplicate action names overwrite each other
**File:** `tools/engine.py:57, 678-693`
**Type:** Logic Bug

In `_parse_setup_robot`, both arm calibrations use `name="calibrate"`:
```python
graph.add_action(ActionPrimitive(name="calibrate", subsystem="left_arm", ...))
graph.add_action(ActionPrimitive(name="calibrate", subsystem="right_arm", ...))
```

But `ActionGraph.add_action` uses name as dict key:
```python
self.actions[action.name] = action  # Second overwrites first!
```

**Impact:** Only right arm calibrates; left arm action lost.

---

### 8. Fire flicker detection broken when smoke disabled
**File:** `visual_safety.py:206-256`
**Type:** Logic Bug

`_prev_gray` (for motion/flicker detection) is only updated in `_detect_smoke()`:
```python
# Line 256 in _detect_smoke:
self._prev_gray = gray
```

If `config.smoke_detection = False`, `_prev_gray` never updates. Fire detection reads stale frame data.

**Impact:** Fire flicker detection fails without smoke detection enabled.

---

### 9. DOA angle wraparound in `_handle_out_of_range`
**File:** `johnny5_body.py:126-131`
**Type:** Logic Bug

The angle comparison logic has issues:
```python
if doa > 90 and doa < 180:
    desc = "behind me to the right"
elif doa >= 180 and doa < 270:
    desc = "behind me to the left"
elif doa < -90 or doa > 270:  # Can never be > 270 after modulo
    desc = "behind me"
```

DOA is 0-360, so `doa < -90` never true, `doa > 270` logic is wrong after earlier conditions.

**Impact:** Direction descriptions incorrect for some angles.

---

### 10. `Subsystem` enum mismatch
**File:** `tools/engine.py:239`, `adapters/base.py`
**Type:** Type Mismatch

In `_execute_wave`, subsystem is a string:
```python
subsystem = "right_arm" if arm == "right" else "left_arm"
```

But later:
```python
subsystem = Subsystem(subsystem_name)  # Expects enum value, not string
```

`Subsystem` enum has values like `LEFT_ARM`, not `"left_arm"`.

**Impact:** Runtime error when executing tools.

---

## Medium: Missing Features

### 11. `detect_robot()` not implemented
**File:** `adapters/__init__.py`
**Type:** Missing Feature

Documentation mentions auto-detecting robot type:
```python
def detect_robot() -> RobotAdapter:
    # Auto-detect based on available hardware/ports
```

But this function doesn't exist in `adapters/__init__.py`.

**Impact:** Can't auto-detect robot platform.

---

### 12. Visual safety not integrated with camera
**File:** `visual_safety.py`, `whoami_full.py`
**Type:** Missing Integration

`VisualSafety.process_frame()` expects frames, but nobody feeds it frames from OAK-D. `whoami_full.py` has the camera but doesn't call visual safety.

**Impact:** Fire/smoke detection never runs on actual camera feed.

---

### 13. Terrain navigation not connected to SLAM/movement
**File:** `terrain_navigation.py`
**Type:** Missing Integration

`TerrainNavigator` has crossing strategies but:
- No integration with the robot's actual movement system
- `_move_callback` never set
- `_speak_callback` never set

**Impact:** Terrain navigation is standalone, not integrated.

---

### 14. Head tracker gantry controller is sync wrapper
**File:** `motion_coordinator.py:166`
**Type:** Potential Issue

```python
self.head_tracker.set_gantry_controller(self._move_gantry_sync)
```

Uses `_move_gantry_sync` (synchronous) but actual movement is async. May cause blocking in tracking thread.

**Impact:** Head tracking may lag or block.

---

### 15. No hitch/lift parsers in engine
**File:** `tools/engine.py`
**Type:** Missing Parsers

`_parse_hitch`, `_parse_tow_cart`, `_parse_dock_charger`, `_parse_calibrate_hitch` not implemented, but tools are registered in `registry.py`.

**Impact:** Hitch-related tools defined but not executable.

---

## Low: Code Quality

### 16. Wrong import in docstring
**File:** `johnny5_body.py:16`
**Type:** Documentation Error

```python
"""
Usage:
    from chloe_body import Johnny5Body
```

Should be:
```python
    from johnny5_body import Johnny5Body
```

**Impact:** Copy-paste from docs fails.

---

### 17. Hardcoded API key in source
**File:** `johnny5.py:35`
**Type:** Security

```python
HUME_API_KEY = os.getenv("HUME_API_KEY", "BAO5bSYoEGCM1hrjCmbd0RseuxKjTuyxok0hEuGpnW7AsH9r")
```

API key in source code, even as fallback.

**Impact:** Key exposed in version control.

---

### 18. Unused imports
**File:** `adapters/johnny5.py:19`
**Type:** Dead Code

```python
import struct  # Not used anywhere in file
```

**Impact:** Minor - clutters imports.

---

### 19. Threading + asyncio mixing
**File:** `head_tracker.py`, `motion_coordinator.py`
**Type:** Architectural

Mixed threading (`threading.Thread`) and asyncio (`async def`) without proper bridges. `_track_loop` runs in thread but needs to call async motor functions.

**Impact:** Potential race conditions, callback issues.

---

### 20. No error handling for Solo-CLI failures
**File:** `motion_coordinator.py:697-707`, `adapters/johnny5.py`
**Type:** Missing Error Handling

```python
result = subprocess.run(cmd, capture_output=True, timeout=timeout)
# No check of result.returncode or stderr
```

Solo-CLI errors silently ignored.

**Impact:** Motor failures not reported.

---

### 21. Registry singleton not thread-safe
**File:** `tools/registry.py:988-993`
**Type:** Thread Safety

```python
_registry = None

def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:  # Race condition window
        _registry = ToolRegistry()
    return _registry
```

No lock around singleton creation.

**Impact:** Multiple registries possible under concurrent access.

---

## Summary

| Category | Count |
|----------|-------|
| Critical (Disconnected) | 6 |
| High (Logic Errors) | 4 |
| Medium (Missing Features) | 5 |
| Low (Code Quality) | 6 |
| **Total** | **21** |

### Priority Order
1. Connect johnny5.py to motion_coordinator (spine integration)
2. Add tool_call handler to on_message
3. Instantiate and connect ToolExecutionEngine
4. Fix duplicate action names in engine.py
5. Call remember_person from face recognition
6. Call process_speech_text and set_speaking from johnny5.py

---

## Part 2: Timing, Race Conditions & Safety Order

Additional issues discovered during timing/concurrency analysis.

---

### 22. `RealTimeCommandQueue` never used
**File:** `tools/realtime.py`
**Type:** Dead Code / Critical for Smoothness

A proper real-time command queue exists with:
- Priority levels (EMERGENCY bypasses everything)
- Timestamp tracking and deadline expiration
- 30Hz control loop with `time.perf_counter()`
- Action interpolation for smooth motion

But **nothing uses it**:
```python
# Exported but never imported anywhere else:
from tools import RealTimeCommandQueue, RealTimeExecutor, get_command_queue
```

**Impact:** No smooth motion interpolation. No command prioritization. No real-time timing.

---

### 23. Fire alert doesn't use EMERGENCY priority
**File:** `motion_coordinator.py:460-490`, `tools/realtime.py`
**Type:** Safety Order Issue

Fire detection triggers `alert_fire()` which does arm movements through regular async calls:
```python
async def alert_fire(self, direction: float = None):
    # Regular async movement - no priority!
    await self._move_arm("left", point_pose, speed=1.0)
```

Should use `CommandPriority.EMERGENCY` from realtime.py to:
1. Interrupt current movement immediately
2. Clear interruptible commands from queue
3. Execute fire alert with highest priority

**Impact:** Fire alert could be delayed by ongoing motion or queued commands.

---

### 24. Multiple independent timing clocks
**File:** Various
**Type:** Timing Inconsistency

Different modules use different timing sources:

| Module | Timing Source | Issue |
|--------|--------------|-------|
| `doa_reader.py` | `time.perf_counter()` | Correct for rate control |
| `head_tracker.py` | `time.time()` + hardcoded `time.sleep(0.03)` | Inconsistent |
| `motion_coordinator.py` | `time.time()` | Not monotonic |
| `johnny5.py` | `time.time()` | For latency logging only |
| `tools/realtime.py` | `time.perf_counter()` | Correct but unused |

`time.time()` can jump backwards (NTP adjustments), `time.perf_counter()` is monotonic.

**Impact:** Rate control may drift. Timestamps may be inconsistent across modules.

---

### 25. Head tracker and motion coordinator lock conflict
**File:** `head_tracker.py:101`, `motion_coordinator.py:158`
**Type:** Potential Deadlock

Both have their own locks:
```python
# head_tracker.py
self._lock = threading.Lock()

# motion_coordinator.py
self._lock = threading.Lock()
```

Head tracker calls motion coordinator's gantry controller from its thread while holding its lock. If motion coordinator tries to read head tracker state while holding its lock, deadlock is possible.

**Impact:** Potential deadlock under specific timing conditions.

---

### 26. Async functions calling blocking subprocess
**File:** `motion_coordinator.py:697`, `adapters/johnny5.py`
**Type:** Blocking in Async Context

```python
async def _move_gantry(self, pan: float, tilt: float, speed: float):
    # This BLOCKS the entire async event loop!
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
```

Should use `asyncio.create_subprocess_exec()` or run in thread pool.

**Impact:** Head tracking blocks all other async operations during motor commands.

---

### 27. No coordination between safety systems
**File:** `visual_safety.py`, `terrain_navigation.py`, `motion_coordinator.py`
**Type:** Safety Priority

Three safety systems that could conflict:
1. **Visual Safety** - fire/smoke detection
2. **Terrain Navigation** - cord/gap detection
3. **Motion Coordinator** - general movement

No coordination layer. If fire is detected while crossing a gap:
- Which takes priority?
- Who controls the motors?
- Could commands conflict?

**Impact:** Undefined behavior when multiple safety conditions trigger simultaneously.

---

### 28. Cooldowns use wall clock time
**File:** `visual_safety.py:303-307`, `motion_coordinator.py:353`
**Type:** Timing Issue

```python
# visual_safety.py
if now - self._last_alert_time < self.config.alert_cooldown_seconds:
    return  # Skip alert!

# motion_coordinator.py
if now - person.last_pointed_at < self._point_cooldown_seconds:
    return  # Don't point
```

Uses `time.time()` which can jump. System clock adjustment could:
- Make cooldown infinite (clock jumped forward)
- Skip cooldown entirely (clock jumped backward)

**Impact:** Safety cooldowns unreliable if system time changes.

---

### 29. Thread-to-async callback bridge missing
**File:** `head_tracker.py:181`, `motion_coordinator.py:166`
**Type:** Threading/Async Issue

```python
# Head tracker runs in thread, calls sync callback:
self.doa.on_direction_change(self._on_doa_change)

# Motion coordinator provides sync wrapper for async:
self.head_tracker.set_gantry_controller(self._move_gantry_sync)
```

But `_move_gantry_sync` needs to call async `_move_gantry`:
```python
def _move_gantry_sync(self, pan, tilt, speed):
    # How to call async from sync thread context?
    # asyncio.run() creates new loop
    # loop.run_until_complete() may not work from thread
```

No proper `asyncio.run_coroutine_threadsafe()` bridge.

**Impact:** Motor commands from head tracker may fail or create race conditions.

---

### 30. LED state machine has no mutex
**File:** `led_controller.py:50,89-175`
**Type:** Race Condition

```python
self._lock = threading.Lock()

def listening(self):
    # Acquires lock, sets pattern

def thinking(self):
    # Acquires lock, sets different pattern
```

But if two threads call different LED states simultaneously, the visual feedback could glitch:
1. Thread A: `listening()` - acquires lock, starts pattern
2. Thread A: releases lock
3. Thread B: `thinking()` - acquires lock, changes pattern
4. Visual: flickers between states

**Impact:** LED feedback may be inconsistent during rapid state changes.

---

### 31. Action interpolation not integrated
**File:** `tools/realtime.py:303-354`
**Type:** Missing Integration

`ActionInterpolator` exists for smooth motion:
```python
class ActionInterpolator:
    """Smooth interpolation between robot positions."""
    def step(self) -> Dict[str, List[float]]:
        # Exponential smoothing toward targets
```

But nothing uses it. Motor commands go directly to Solo-CLI without interpolation.

**Impact:** Jerky motion. No smooth transitions between poses.

---

### 32. No heartbeat/watchdog for motor safety
**File:** `adapters/johnny5.py`, `motion_coordinator.py`
**Type:** Safety

No watchdog that:
- Checks if motor commands are still being sent
- Disables torque if communication lost
- Detects if control loop stopped

If the control program crashes, motors stay in last position with torque enabled.

**Impact:** Robot could be stuck in position after crash, requiring manual intervention.

---

### 33. Gesture interruption not handled
**File:** `motion_coordinator.py:255-290`
**Type:** State Machine Issue

During gestures like `wave()` or `express_excitement()`:
```python
async def wave(self, arm: str = "right", style: str = "friendly"):
    await self._gesture(Gesture.WAVE, arm=arm)
    # What if fire detected here?
    await asyncio.sleep(0.2)
    # Gesture continues even during emergency!
```

No mechanism to interrupt mid-gesture for higher priority actions.

**Impact:** Emergency responses delayed until gesture completes.

---

## Updated Summary

| Category | Count |
|----------|-------|
| Critical (Disconnected) | 6 |
| High (Logic Errors) | 4 |
| Medium (Missing Features) | 5 |
| Low (Code Quality) | 6 |
| **Timing/Safety (New)** | **12** |
| **Total** | **33** |

### Timing/Safety Priority Order
1. Integrate `RealTimeCommandQueue` for all motor commands
2. Use `CommandPriority.EMERGENCY` for fire alerts
3. Add proper async/thread bridge using `run_coroutine_threadsafe()`
4. Standardize on `time.perf_counter()` for all timing
5. Add safety coordination layer for multi-system conflicts
6. Integrate `ActionInterpolator` for smooth motion
7. Add motor watchdog/heartbeat
