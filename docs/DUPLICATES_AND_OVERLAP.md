# Duplicate Logic and Overlapping Systems

Survey of redundant implementations in the Johnny Five codebase.

---

## ✅ CONSOLIDATION COMPLETE

The following consolidation has been applied:

| Issue | Solution | Status |
|-------|----------|--------|
| Motor config (3 sources) | `config/hardware.py` is now single source | ✅ Done |
| Motor ping (3 implementations) | `config/motors.py:MotorInterface.ping()` | ✅ Done |
| Calibration systems | YAML defines actions, Python executes | ✅ Done |
| Startup calibration | `actions/startup_calibration.yaml` + motor interface | ✅ Done |

**Files created:**
- `config/__init__.py` - Package init
- `config/hardware.py` - `HardwareConfig` single source of truth
- `config/motors.py` - `MotorInterface` consolidated operations
- `actions/startup_calibration.yaml` - Movie-style boot sequence

**Files updated to use shared config:**
- `motion_coordinator.py` - imports from `config.hardware`
- `adapters/johnny5.py` - imports from `config.hardware`
- `tools/verbal_calibration.py` - imports from `config.hardware` and `config.motors`

---

## Historical: Motor Configuration (3 Sources of Truth)

The same hardware layout is defined in THREE places:

| Location | Format | Lines |
|----------|--------|-------|
| `motion_coordinator.py:57-80` | Python `MotorConfig` class | 23 |
| `adapters/johnny5.py:30-50` | Python `Johnny5HardwareConfig` class | 20 |
| `actions/servo_banks.yaml` | YAML config | 150 |

All three define:
```
ACM0: left_arm (1-6), wheels (7-9), lift (10)
ACM1: right_arm (1-6), gantry (7-8)
```

**Problem**: Change hardware layout = update 3 files.

---

## High: Motor Ping (3 Implementations)

The same "ping a motor" operation exists in three places:

| File | Function | Returns |
|------|----------|---------|
| `tools/verbal_calibration.py:235-249` | `_ping_motor()` | `bool` |
| `motion_coordinator.py:541-571` | `_ping_servo()` | `ServoHealth` |
| `adapters/johnny5.py:504-529` | `scan_motors()` | `Dict` |

All run the same command:
```bash
solo robo --port {{port}} --ids {{id}} --ping
```

---

## High: Calibration Systems (3 Approaches)

Three different ways to calibrate:

### 1. Verbal Calibration (`tools/verbal_calibration.py`)
```python
class VerbalCalibration:
    # Interactive, human-guided
    # "I found 10 motors on ACM0..."
    # Uses speech + human confirmation

    expected_layout = {
        "ACM0": {"left_arm": [1-6], "base": [7-9], "lift": [10]},
        "ACM1": {"right_arm": [1-6], "gantry": [7,8]}
    }
```

### 2. Adapter Calibration (`adapters/johnny5.py`)
```python
class Johnny5Adapter:
    # Programmatic, four methods
    calibrate_arm(side, mode)    # "quick" | "full" | "offsets_only"
    calibrate_gantry(mode)
    calibrate_lift(mode)
    calibrate_base(mode)
```

### 3. YAML Actions (`actions/gantry_calibration.yaml`)
```yaml
# Declarative sequence
execution:
  steps:
    - name: center_gantry
      action: move_to_position
      speak: "Centering the camera gantry"
```

**Conflict**: Each has different execution model, different state storage.

---

## Medium: Gantry/Camera Control (2 Paths)

| File | Method | Interface |
|------|--------|-----------|
| `motion_coordinator.py:685-709` | `_move_gantry(pan, tilt, speed)` | Raw angles |
| `adapters/johnny5.py:447-474` | `_look_at(params)` | Named targets |

~20 lines of duplicate code for the same Solo-CLI command.

---

## Medium: Wave Gesture (2 Implementations)

| File | Method | Lines |
|------|--------|-------|
| `motion_coordinator.py:827-845` | `_do_wave()` | 18 |
| `adapters/johnny5.py:410-445` | `_wave_action()` | 35 |

Same arm motion sequence, different wrappers.

---

## Medium: Fire/Safety Alert (Disconnected)

Two systems that should be integrated:

### Detection (`visual_safety.py`)
```python
def _detect_fire(frame):
    # HSV color detection
    # Returns HazardDetection with confidence + DOA
```

### Response (`motion_coordinator.py`)
```python
async def alert_fire(direction: float):
    # Both arms point at fire
    # Twitch for attention
```

**Problem**: No integration. Detection doesn't trigger response automatically.

---

## Medium: Identity Storage (2 Systems)

| System | File | Storage | What it stores |
|--------|------|---------|----------------|
| Self-Identity | `tools/self_identity.py` | `~/whoami/self_identity.pkl` | Color histogram, contours |
| Face Recognition | `whoami_full.py` | `~/whoami/last_seen.pkl` | Face embeddings, timestamps |

Neither integrates with MemoRable for long-term memory.

---

## Low: Tool Registration (Split)

Tools defined in two places:

| File | What | Count |
|------|------|-------|
| `tools/registry.py:52-954` | Tool definitions (JSON schema) | ~45 tools |
| `tools/engine.py:111-141` | Tool parsers | ~45 parsers |

Must register in BOTH places to add a new tool.

---

## Summary Table

| System | Files | Duplicates | Severity |
|--------|-------|------------|----------|
| Motor Config | 3 files | 3 definitions | CRITICAL |
| Motor Ping | 3 files | 3 implementations | HIGH |
| Calibration | 3 files | 3 approaches | HIGH |
| Gantry Control | 2 files | ~20 duplicate lines | MEDIUM |
| Arm Gestures | 2 files | ~40 duplicate lines | MEDIUM |
| Fire Safety | 2 files | Not integrated | MEDIUM |
| Identity | 2 files | 2 storage systems | MEDIUM |
| Tool System | 2 files | Split registration | LOW |

---

## Consolidation Plan

### Phase 1: Single Hardware Config

Create `config/hardware.py`:
```python
@dataclass
class HardwareConfig:
    """Single source of truth for motor layout."""

    ACM0_PORT = "/dev/ttyACM0"
    ACM1_PORT = "/dev/ttyACM1"

    LEFT_ARM_IDS = (1, 2, 3, 4, 5, 6)
    RIGHT_ARM_IDS = (1, 2, 3, 4, 5, 6)
    WHEEL_IDS = (7, 8, 9)
    LIFT_ID = 10
    GANTRY_IDS = (7, 8)
```

Remove: `MotorConfig` from `motion_coordinator.py`, `Johnny5HardwareConfig` from `johnny5.py`

### Phase 2: Motor Interface

Create `hardware/motor_interface.py`:
```python
class MotorInterface:
    """Single implementation for motor operations."""

    async def ping(port, motor_id) -> bool
    async def scan(port, id_range) -> Dict
    async def wiggle(port, motor_id) -> bool
    async def move(port, motor_id, position) -> bool
```

Used by: `verbal_calibration.py`, `motion_coordinator.py`, `johnny5.py`

### Phase 3: Unified Calibration

Keep `VerbalCalibration` as orchestrator, use `MotorInterface` for primitives.

Delete: `actions/gantry_calibration.yaml` (absorbed into verbal system)

### Phase 4: Integrate Safety

```python
# visual_safety.py detects
hazard = visual_safety.process_frame(frame)

# motion_coordinator.py responds
if hazard and hazard.type == HazardType.FIRE:
    await motion_coordinator.alert_fire(hazard.direction)
```

### Phase 5: Unified Identity

Store all identity data in MemoRable:
- Face embeddings
- Voice embeddings
- Self-recognition features
- Stylometry patterns

Delete: `~/whoami/*.pkl` files

---

## Lines to Eliminate

| Area | Current Lines | After Consolidation | Saved |
|------|---------------|---------------------|-------|
| Motor Config | ~50 | ~20 | 30 |
| Motor Ping | ~90 | ~30 | 60 |
| Calibration | ~600 | ~400 | 200 |
| Gantry | ~50 | ~30 | 20 |
| Gestures | ~75 | ~40 | 35 |
| Identity | ~200 | ~120 | 80 |
| **Total** | **~1065** | **~640** | **~425** |

**40% reduction** in duplicated code.
