# Sensor Utilization Report

Current usage vs available capabilities for Johnny Five's primary sensors.

---

## ReSpeaker USB Mic Array (4-Mic Far-Field)

**Hardware:** Seeed ReSpeaker with XMOS XVF-3000 DSP

### What We're Using

| Feature | File | Status |
|---------|------|--------|
| DOA (Direction of Arrival) | `doa_reader.py` | ✅ Full |
| LED Ring (12 RGB) | `led_controller.py` | ✅ Full |
| Hardware VAD | `doa_reader.py` | ✅ Basic |
| 1-Channel Processed Audio | `johnny5.py` | ✅ Working |
| Head Tracking via DOA | `head_tracker.py` | ✅ Working |

### What's Available But Unused

| Feature | Capability | Current State | Value |
|---------|------------|---------------|-------|
| **6-Channel Raw Audio** | 4 raw mics + 2 processed | Using 1-channel only | HIGH - Custom beamforming, ML training |
| **Speech Probability** | 0-255 confidence from XMOS | Read but not used | MEDIUM - Better turn-taking |
| **Per-Mic Levels** | Individual mic gain readings | Read but not used | LOW - Debugging only |
| **Custom Beamforming** | Steerable audio focus | Using default | MEDIUM - Focus on speaker |
| **AEC Reference Signal** | Echo cancellation input | Not connected | HIGH - Better duplex |

### Recommendations

1. **Connect AEC Reference** (Priority: HIGH)
   - Route speaker audio through ReSpeaker 3.5mm jack
   - OR use PulseAudio loopback to XMOS
   - Eliminates need for software muting

2. **Use Speech Probability for Turn-Taking** (Priority: MEDIUM)
   - `doa_reader.read_speech_probability()` returns 0.0-1.0
   - Use to detect when human stops speaking
   - Smoother conversation flow

3. **Consider 6-Channel Mode** (Priority: LOW for hackathon)
   - Requires firmware flash
   - Enables custom SRP-PHAT DOA via pyroomacoustics
   - Better for post-hackathon ML work

---

## OAK-D Pro Camera

**Hardware:** Luxonis OAK-D Pro with stereo depth + RGB

### What We're Using

| Feature | File | Status |
|---------|------|--------|
| RGB Camera (640x480) | `whoami_full.py` | ✅ Working |
| YOLO Detection | `whoami_full.py` | ✅ Every 8 frames |
| Face Recognition | `whoami_full.py` | ✅ Working |
| Gantry Control | `head_tracker.py` | ✅ Pan/Tilt tracking |

### What's Available But Unused

| Feature | Capability | Current State | Value |
|---------|------------|---------------|-------|
| **Depth Sensing** | Stereo depth map | Not used | HIGH - Spatial awareness |
| **Spatial Detection** | 3D bounding boxes (X,Y,Z) | Not used | HIGH - Know distance to people |
| **On-device Neural** | Run YOLO on VPU | Running on Jetson CPU | MEDIUM - Lower latency |
| **IMU Data** | Accelerometer/Gyro | Not used | LOW - Motion detection |
| **Higher Resolution** | Up to 4K RGB | Using 640x480 | LOW - Recognition works fine |
| **IR Illumination** | Night vision (Pro model) | Not used | MEDIUM - Low light |

### Current Gaps

1. **Visual Safety Not Fed Frames** (documented in BUGS_AND_IMPROVEMENTS.md)
   - `visual_safety.py` has fire/smoke detection
   - `whoami_full.py` has camera but doesn't call it
   - **Fix:** Add `visual_safety.process_frame(frame)` to camera loop

2. **No Depth Data for Terrain**
   - `terrain_navigation.py` mentions "needs depth sensing"
   - Could detect: stairs, curbs, cords, obstacles
   - **Fix:** Enable stereo depth pipeline

3. **YOLO Running on CPU**
   - Could run on OAK-D's VPU (Myriad X)
   - Lower latency, frees Jetson GPU
   - **Fix:** Use depthai neural inference

### Recommendations

1. **Enable Spatial Detection** (Priority: HIGH)
   ```python
   # Get 3D position of detected faces
   spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
   # Returns: x, y, z coordinates in millimeters
   ```
   - Know exactly how far each person is
   - Better DOA-to-person fusion (depth confirms identity)

2. **Feed Visual Safety** (Priority: HIGH)
   ```python
   # In whoami_full.py camera loop:
   from visual_safety import get_visual_safety
   safety = get_visual_safety()
   safety.process_frame(frame)  # Check for fire/hazards
   ```

3. **Move YOLO to VPU** (Priority: MEDIUM)
   - Compile model for Myriad X
   - Use `depthai` built-in object detection
   - Frees Jetson for other tasks

4. **Add Depth for Navigation** (Priority: LOW for hackathon)
   - Enable stereo depth node
   - Detect obstacles in path
   - Guide dog mode enhancement

---

## Fusion: ReSpeaker + OAK-D

### Currently Integrated

| Integration | Implementation | File |
|-------------|----------------|------|
| DOA → Gantry Pan | DOA angle maps to servo position | `head_tracker.py` |
| DOA + YOLO → Active Speaker | Gaussian scoring by pixel position | `docs/compass_artifact*.md` |

### Missing Integration

| Integration | Status | Value |
|-------------|--------|-------|
| **DOA + Depth → Person Distance** | Not implemented | Confirm speaker identity by position |
| **Voice ID + Face ID → Multimodal** | Documented, not implemented | Robust identification |
| **LED + Detection → Visual Feedback** | Partial | Show who robot is attending to |

### Fusion Architecture (Target)

```
ReSpeaker                          OAK-D Pro
    │                                  │
    ├─ DOA (direction)                 ├─ RGB (faces)
    ├─ VAD (speaking?)                 ├─ Depth (distance)
    ├─ Voice embedding                 ├─ YOLO (detection)
    │                                  │
    └──────────┬───────────────────────┘
               │
        ┌──────▼──────┐
        │   FUSION    │
        │             │
        │ DOA angle ──┼── Pixel column
        │ Distance ───┼── Depth at bbox
        │ Voice ID ───┼── Face ID
        │             │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  IDENTITY   │
        │             │
        │ High conf: face+voice+DOA agree
        │ Med conf: face+DOA agree
        │ Low conf: voice only (blindfolded)
        └─────────────┘
```

---

## Quick Wins (Do Today)

1. **Visual Safety Integration** - 10 lines of code
   - Add `safety.process_frame(frame)` to `whoami_full.py`
   - Fire detection starts working immediately

2. **Speech Probability for Attention** - Already reading it
   - Use `speech_prob` from `get_doa()` return value
   - Better indicator than binary VAD

3. **LED Shows Attention Target** - Already have LED control
   - Point LEDs toward DOA direction (mode 4 does this)
   - Different color when person identified

---

## Summary

| Sensor | Utilization | Status |
|--------|-------------|--------|
| **ReSpeaker** | ~70% | Connect AEC reference (remaining) |
| **OAK-D Pro** | ~70% | ✅ Spatial detection + visual safety integrated |
| **Fusion** | ~80% | ✅ DOA+Depth person matching implemented |

### Recent Improvements

1. **Spatial Detection** (`spatial_tracker.py`)
   - Stereo depth pipeline enabled
   - 3D person positions (x, y, z in mm)
   - Angle and distance from camera center
   - `get_person_at_angle()` for DOA fusion

2. **DOA + Spatial Fusion** (`doa_spatial_fusion.py`)
   - Combines ReSpeaker DOA with OAK-D spatial detection
   - Gaussian scoring for angular match
   - Depth confidence weighting
   - Active speaker identification in multi-person scenes

3. **Visual Safety Integration**
   - `spatial_tracker.enable_visual_safety()` feeds frames to hazard detection
   - Fire and smoke detection via `visual_safety.py`
   - Runs automatically with camera loop

### Remaining Work

1. **AEC Reference** (Priority: HIGH)
   - Route speaker audio through ReSpeaker for echo cancellation

2. **Move YOLO to VPU** (Priority: MEDIUM)
   - Currently runs on Jetson CPU
   - Could run on OAK-D's Myriad X for lower latency

3. **Voice + Face Identity Fusion** (Priority: MEDIUM)
   - DOA+Depth now identifies WHO is speaking
   - Next: Link to face/voice embeddings for named identity
