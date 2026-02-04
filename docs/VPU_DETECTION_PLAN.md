# VPU Detection Plan

## Goal
Load YuNet face detection on VPU via Luxonis Model ZOO. Zero CPU load for detection.

## Architecture
```
OAK-D Camera → YuNet (VPU) → Face Detections → IPC Bus → johnny5.py
                                    ↓
                            CentroidTracker → Face Recognition → Greeting
```

## Implementation
1. Use `dai.NNModelDescription("luxonis/yunet:640x480")`
2. Use `ParsingNeuralNetwork` from `depthai_nodes`
3. Output: `ImgDetectionsExtended` with face bboxes + keypoints
4. Publish detections via IPC to johnny5.py

## Current Status
- YuNet model loads successfully on VPU
- OpenCV GUI crashed (headless package conflict)
- Need to fix OpenCV and test full pipeline

## Next Step
Run on Jetson with display connected.
