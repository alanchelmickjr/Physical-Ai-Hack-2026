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
