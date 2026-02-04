# People Detection Fix - Crash & "One Person" Bug

**Problem:** Robot crashes when overwhelmed by many people. All detections merge into one person.

---

## Root Cause Analysis

### Issue 1: Blob Path Mismatch (VPU Failure)

`whoami_full.py:378-386` looks for:
```python
blob_paths = [
    os.path.expanduser("~/models/yolov8n_coco_640x352.blob"),
    os.path.expanduser("~/yolov8n_coco_640x352.blob"),
]
```

But the repo has `models/yolov8n_coco_640x480.blob` - **wrong dimensions**.

**Result:** VPU YOLO fails silently, falls back to CPU YOLO. CPU YOLO is slower and may not handle crowds.

### Issue 2: CentroidTracker Merges Close People

`whoami_full.py:252-254`:
```python
# Max distance threshold (300 pixels - allow for movement)
if D[row, col] > 300:
    continue
```

When people are standing close together (hackathon crowd), centroids are within 300px and get matched to the same track_id. All become "one person".

### Issue 3: No Multi-Person Announcement Queue

When 5+ people walk in together:
1. All get same track_id (Issue 2)
2. Only one gets announced
3. Rest are "already announced" (line 757-763)

### Issue 4: Face Recognition Race

`whoami_full.py:750-754`:
```python
if person.needs_recognition():
    face_h = int((y2 - y1) * 0.4)
    face_bbox = (x1, y1, x2, y1 + face_h)
    name, conf = self.recognize_face(frame, face_bbox)
```

With multiple people, face recognition runs on wrong bbox (merged track).

---

## Fix Plan

### Step 1: Fix Blob Path

Update `whoami_full.py` to look for the actual blob:
```python
blob_paths = [
    os.path.expanduser("~/models/yolov8n_coco_640x480.blob"),
    "models/yolov8n_coco_640x480.blob",  # Local repo path
]
```

Also update pipeline camera size to match (640x480).

### Step 2: Reduce Centroid Distance Threshold

Lower from 300px to 150px to prevent merging close people:
```python
if D[row, col] > 150:
    continue
```

### Step 3: Add Detection Logging

Log raw YOLO detection count before tracking:
```python
logger.info(f"YOLO detections: {len(bboxes)} people")
```

Log tracker state:
```python
logger.info(f"Tracker: {len(tracked_persons)} tracks, IDs: {[p.track_id for p in tracked_persons]}")
```

### Step 4: Clear Stale Tracks Faster

Reduce `max_disappeared` from 60 to 30 frames:
```python
self.tracker = CentroidTracker(max_disappeared=30)
```

---

## Testing Checklist

- [ ] VPU mode activates (check "VPU YOLO started" in logs)
- [ ] Multiple people get separate track_ids
- [ ] Each person announced individually
- [ ] No crashes with 5+ people
- [ ] Face recognition matches correct person

---

## Files to Modify

1. `whoami_full.py` - Blob paths, threshold, logging
2. (optional) Delete mismatched blobs to force re-download

---

## DepthAI 3.x Compatibility Issue

**Status:** VPU YOLO fails, CPU fallback works

The robot has depthai 3.x installed, but the code uses 2.x API:
```
WARNING: VPU YOLO failed: module 'depthai.node' has no attribute 'XLinkOut'
DeprecationWarning: ColorCamera node is deprecated. Use Camera node instead.
```

### Differences in depthai 3.x

| 2.x API | 3.x API |
|---------|---------|
| `dai.node.XLinkOut` | `queue = output.createOutputQueue()` |
| `dai.node.ColorCamera` | `dai.node.Camera` |
| `device.getOutputQueue()` | `output.createOutputQueue()` |

### Future Fix

Update `_try_vpu_yolo()` to use depthai 3.x API. For now, CPU YOLO fallback is working.

---

## Applied Fixes (2026-02-03)

1. **Camera preview size:** Changed from 640x480 to 640x352 to match blob
2. **Blob paths:** Updated to look for correct filename
3. **Centroid threshold:** Reduced from 300px to 150px
4. **Max disappeared:** Reduced from 60 to 30 frames
5. **Logging:** Added detection count and track ID logging

---

## VPU Detection via HubAI Model Zoo (2026-02-03)

**MAJOR FIX:** Removed all CPU/GPU YOLO. Now using pure VPU detection via DepthAI V3 + HubAI Model Zoo.

### New Architecture

```python
from depthai_nodes.node import ParsingNeuralNetwork

# HubAI model - auto-downloaded, runs 100% on VPU
model = "luxonis/yolov8-nano:coco-320x320"

cam = pipeline.create(dai.node.Camera).build()
nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(cam, model)
detection_queue = nn_with_parser.out.createOutputQueue()
```

### Benefits

1. **Zero CPU/GPU load** - All inference on Myriad X VPU
2. **Auto model download** - HubAI handles model caching
3. **Proper V3 API** - No deprecated XLinkOut/ColorCamera
4. **Handles crowds** - VPU can run full speed without thermal throttle

### Install Requirement

```bash
pip install depthai-nodes
```

### Fallback Model

If YOLOv8-nano doesn't work, try YuNet face detection:
```python
model = "luxonis/yunet:640x480"
```

---

## Legacy Blob Files (Can Delete)

```
models/yolov8n_coco_640x480.blob  # No longer used - HubAI handles models
models/yolov5_416.blob            # Legacy
models/mobilenet-ssd.blob         # Legacy
```
