# Audio-Visual Fusion for Speaker Localization on Social Robots

A **ReSpeaker 4-mic array combined with OAK-D YOLO detection** can reliably identify which person is speaking by fusing Direction of Arrival (DOA) audio estimates with visual bounding boxes. The core technique maps the DOA angle to camera pixel coordinates, then scores each detected person based on angular proximity to the sound source. For a hackathon build, expect **±10° DOA accuracy** and **50-100ms end-to-end latency** using the recommended SRP-PHAT algorithm with probabilistic fusion—sufficient for natural conversational interaction.

This document provides the complete technical architecture for Johnny Five, including coordinate transformation math, code examples, and a ROS node template for real-time operation.

---

## How 4-mic arrays estimate sound direction

Directional microphone arrays determine sound source direction by measuring **Time Difference of Arrival (TDOA)** between microphone pairs. When sound reaches one microphone before another, the delay encodes the source angle. A 4-microphone circular array (typical radius: **50-100mm**) provides sufficient geometry to resolve azimuth angles in a 360° plane.

Three algorithms dominate the field, each with distinct tradeoffs:

| Algorithm | Best For | Latency | Accuracy (4-mic) |
|-----------|----------|---------|------------------|
| **GCC-PHAT** | Single speaker, speed | ~16-32ms | ±10-15° |
| **MUSIC** | Multiple sources, precision | ~40-60ms | ±1-5° |
| **SRP-PHAT** | Reverberant rooms, robustness | ~35-50ms | ±5-10° |

**SRP-PHAT is the recommended choice** for social robot applications. It combines the robustness of phase-transform weighting with spatial grid search, accumulating evidence across all microphone pairs. Unlike GCC-PHAT (single-source assumption) or MUSIC (requires knowing source count), SRP-PHAT handles typical indoor environments with moderate reverberation while maintaining real-time performance.

The core formula steers a virtual beamformer across candidate angles and finds where power is maximized:

```
P(θ) = Σ GCC-PHAT_{m1,m2}(τ(θ))  for all mic pairs (m1, m2)
```

**Pyroomacoustics** provides production-ready implementations:

```python
import numpy as np
import pyroomacoustics as pra

# 4-mic circular array (50mm radius)
radius = 0.05
angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
mic_positions = np.array([
    [radius * np.cos(a), radius * np.sin(a), 0] 
    for a in angles
]).T

# SRP-PHAT DOA estimator
doa = pra.doa.SRP(
    L=mic_positions,
    fs=16000,
    nfft=512,  # ~32ms frame
    c=343.0,
    num_src=1,
    azimuth=np.linspace(-np.pi, np.pi, 360)
)

# Process STFT of multichannel audio
X = pra.transform.stft.analysis(audio, 512, 256).transpose([2, 1, 0])
doa.locate_sources(X, freq_range=[500, 4000])
estimated_angle = np.degrees(doa.azimuth_recon[0])
```

**Key parameters**: Use **16kHz sample rate**, **512-sample FFT** (32ms frames), and restrict frequency analysis to **500-4000Hz** (speech band). For a 50mm radius array, spatial aliasing occurs above ~2400Hz, so higher frequencies should be excluded for clean DOA estimates.

---

## ReSpeaker provides the fastest path to working DOA

The **ReSpeaker Mic Array v2.0** ($70, Seeed Studio) is the optimal hardware choice for hackathon speed. It features 4 MEMS microphones, an XMOS XVF-3000 DSP that handles DOA onboard, and USB connectivity requiring no additional hardware.

The device exposes DOA directly via USB HID—no audio processing code required:

```python
from tuning import Tuning  # from respeaker/usb_4_mic_array repo
import usb.core

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
mic = Tuning(dev)

while True:
    angle = mic.direction      # 0-359 degrees
    voice_active = mic.is_voice()  # Built-in VAD
    print(f"DOA: {angle}°, Speaking: {voice_active}")
```

For higher accuracy, pipe raw audio through **ODAS** (Open embeddeD Audition System), which provides professional-grade localization and tracking:

```bash
# Install ODAS
git clone https://github.com/introlab/odas.git
cd odas && mkdir build && cd build && cmake .. && make

# Run with ReSpeaker config
./bin/odaslive -c respeaker_usb_4_mic_array.cfg
```

ODAS outputs tracked sources via TCP socket in JSON format:
```json
{
  "src": [{
    "id": 1,
    "x": 0.707, "y": 0.707, "z": 0.0,
    "activity": 0.95
  }]
}
```

Convert to angle: `azimuth = atan2(y, x) * 180/π`

| Hardware | DOA Method | Latency | Best For |
|----------|-----------|---------|----------|
| ReSpeaker v2.0 | Built-in DSP | 8-16ms | Simplicity |
| ReSpeaker + ODAS | Software | 20-40ms | Accuracy |
| Matrix Voice | ODAS required | 30-50ms | 8-mic resolution |
| miniDSP UMA-8 | Built-in DSP | ~10ms | Plug-and-play |

---

## Mapping DOA angle to camera bounding boxes

The critical fusion step converts the DOA azimuth angle to a pixel x-coordinate, enabling comparison with YOLO detection bounding boxes. For an OAK-D camera with the microphone array mounted concentrically (same center point), the transformation uses the pinhole camera model:

```python
def doa_to_pixel_x(doa_angle_deg, camera_hfov=68.8, image_width=640):
    """
    Convert DOA azimuth to image x-coordinate.
    
    Args:
        doa_angle_deg: Azimuth from mic array (0° = forward, + = right)
        camera_hfov: OAK-D RGB horizontal FOV in degrees
        image_width: Image width in pixels
    """
    cx = image_width / 2  # Principal point (image center)
    focal_length_px = image_width / (2 * np.tan(np.radians(camera_hfov / 2)))
    
    x_pixel = cx + focal_length_px * np.tan(np.radians(doa_angle_deg))
    return x_pixel
```

For OAK-D at 640×400 resolution (HFOV ≈ 68.8°): **focal_length ≈ 485 pixels**. A DOA of +15° maps to pixel x ≈ 450.

**Important calibration notes**:
- Ensure mic array and camera share the same "forward" direction (0° azimuth = camera optical axis)
- If offset, add a fixed rotation: `doa_corrected = doa_raw + mount_offset_degrees`
- OAK-D provides factory calibration via `device.readCalibration()`—use this for precise intrinsics

---

## Probabilistic scoring identifies the speaker

With DOA mapped to pixel coordinates and YOLO providing person bounding boxes, scoring determines which detected person is most likely speaking. The algorithm computes **Gaussian likelihood** based on angular distance:

```python
def score_speaker_candidates(doa_angle, detections, vad_probability, 
                             camera_params, doa_sigma=10.0):
    """
    Score each detected person as potential speaker.
    
    Args:
        doa_angle: Estimated DOA in degrees
        detections: List of YOLO detections [(bbox, confidence), ...]
        vad_probability: Voice Activity Detection probability [0-1]
        camera_params: (focal_length_px, cx)
        doa_sigma: DOA uncertainty in degrees (±10° typical for 4-mic)
    """
    if vad_probability < 0.3:
        return None  # No one speaking
    
    focal_length, cx = camera_params
    scores = []
    
    for bbox, det_conf in detections:
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2
        
        # Convert bbox center to angle
        bbox_angle = np.degrees(np.arctan((bbox_center_x - cx) / focal_length))
        
        # Gaussian likelihood: how well does DOA match this person?
        angular_distance = abs(doa_angle - bbox_angle)
        doa_likelihood = np.exp(-0.5 * (angular_distance / doa_sigma)**2)
        
        # Combined score
        score = doa_likelihood * det_conf * vad_probability
        scores.append({
            'bbox': bbox,
            'angular_distance': angular_distance,
            'score': score
        })
    
    # Normalize to probabilities
    total = sum(s['score'] for s in scores)
    for s in scores:
        s['probability'] = s['score'] / total if total > 0 else 0
    
    return max(scores, key=lambda x: x['score'])
```

**Handling ambiguous cases** (DOA points between two people):
- Apply **temporal consistency prior**: Previous speaker gets 20-30% bonus (people tend to continue speaking)
- Use **lip movement detection** via MediaPipe Face Mesh as tie-breaker
- Fall back to **maximum likelihood** when uncertain

The expected accuracy with a 4-mic array: if two people are separated by more than **20-25°** (roughly 1 meter apart at 3 meters distance), the system reliably distinguishes them.

---

## Temporal alignment keeps audio and video synchronized

Audio DOA and video frames must be time-aligned for accurate fusion. The target tolerance is **±40ms**—the ITU perceptual threshold for audio-visual synchronization.

```python
from collections import deque
import threading

class AudioVideoSynchronizer:
    def __init__(self, audio_sr=16000, video_fps=30):
        self.audio_sr = audio_sr
        self.video_frame_ms = 1000 / video_fps  # 33.3ms at 30fps
        
        # Circular buffers with timestamps
        self.audio_buffer = deque(maxlen=int(audio_sr * 0.5))  # 500ms
        self.video_buffer = deque(maxlen=15)  # 500ms at 30fps
        self.lock = threading.Lock()
    
    def add_audio(self, samples, timestamp_ms):
        with self.lock:
            self.audio_buffer.append((samples, timestamp_ms))
    
    def add_video(self, frame, detections, timestamp_ms):
        with self.lock:
            self.video_buffer.append((frame, detections, timestamp_ms))
    
    def get_aligned_pair(self):
        """Return closest audio-video pair within tolerance."""
        with self.lock:
            if not self.audio_buffer or not self.video_buffer:
                return None
            
            video_frame, detections, video_ts = self.video_buffer[-1]
            
            # Find audio closest to video timestamp
            best_audio, best_diff = None, float('inf')
            for audio, audio_ts in self.audio_buffer:
                diff = abs(audio_ts - video_ts)
                if diff < best_diff:
                    best_audio, best_diff = audio, diff
            
            if best_diff <= 40:  # Within 40ms tolerance
                return (best_audio, video_frame, detections)
            return None
```

**Threading architecture**: Run audio capture and DOA estimation in a dedicated thread (16kHz, continuous), video/YOLO in another thread (30fps), and fusion in a timer-driven callback (100Hz). Use locks for thread-safe data sharing.

---

## Voice Activity Detection filters when to process

**Silero VAD** provides the best accuracy for open-source VAD, with 87.7% true positive rate at 5% false positive rate:

```python
import torch

model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')

def process_chunk(audio_chunk):
    """Return speech probability [0-1]."""
    tensor = torch.tensor(audio_chunk, dtype=torch.float32)
    return model(tensor, 16000).item()
```

Only run DOA estimation and fusion when `speech_probability > 0.5`. This reduces computation and eliminates false speaker assignments during silence.

---

## Complete ROS2 fusion node for Johnny Five

This template integrates all components into a ROS2 node compatible with OAK-D and ReSpeaker:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Int32
import numpy as np
import threading

class SpeakerFusionNode(Node):
    def __init__(self):
        super().__init__('speaker_fusion')
        
        # Subscribers
        self.create_subscription(Int32, '/sound_direction', self.doa_callback, 10)
        self.create_subscription(Detection2DArray, '/oak/nn/detections', 
                                 self.detection_callback, 10)
        
        # Publisher: identified speaker
        self.speaker_pub = self.create_publisher(PoseStamped, '/speaker_target', 10)
        
        # Camera parameters (OAK-D RGB, 640px width)
        self.focal_length = 485.0
        self.cx = 320.0
        self.doa_sigma = 10.0
        
        # Thread-safe state
        self.lock = threading.Lock()
        self.current_doa = None
        self.current_detections = []
        
        # Fusion timer (30Hz)
        self.create_timer(0.033, self.fusion_callback)
    
    def doa_callback(self, msg):
        with self.lock:
            # Convert 0-359 to -180 to +180 (0 = forward)
            angle = msg.data
            if angle > 180:
                angle -= 360
            self.current_doa = angle
    
    def detection_callback(self, msg):
        with self.lock:
            self.current_detections = [
                (d.bbox, d.results[0].score) 
                for d in msg.detections
                if d.results and d.results[0].id == 'person'
            ]
    
    def fusion_callback(self):
        with self.lock:
            if self.current_doa is None or not self.current_detections:
                return
            
            doa = self.current_doa
            detections = self.current_detections
        
        # Score each person
        best_score, best_bbox = 0, None
        for bbox, conf in detections:
            center_x = (bbox.center.x)
            person_angle = np.degrees(np.arctan((center_x - self.cx) / self.focal_length))
            
            angular_dist = abs(doa - person_angle)
            likelihood = np.exp(-0.5 * (angular_dist / self.doa_sigma)**2)
            score = likelihood * conf
            
            if score > best_score:
                best_score, best_bbox = score, bbox
        
        if best_bbox and best_score > 0.3:
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_link'
            msg.pose.position.x = float(best_bbox.center.x)
            msg.pose.position.y = float(best_bbox.center.y)
            self.speaker_pub.publish(msg)
            self.get_logger().info(f'Speaker at pixel ({best_bbox.center.x:.0f}, {best_bbox.center.y:.0f})')

def main():
    rclpy.init()
    rclpy.spin(SpeakerFusionNode())
    rclpy.shutdown()
```

**Launch commands**:
```bash
# Terminal 1: OAK-D with person detection
ros2 launch depthai_ros_driver camera.launch.py

# Terminal 2: ReSpeaker DOA
ros2 run respeaker_ros respeaker_node

# Terminal 3: Fusion
ros2 run johnny_five speaker_fusion
```

---

## Latency budget achieves real-time performance

For natural interaction, target **<100ms** end-to-end latency:

| Stage | Latency | Notes |
|-------|---------|-------|
| Audio capture buffer | 16ms | 256 samples @ 16kHz |
| DOA estimation | 20-40ms | SRP-PHAT on 512-sample frame |
| Video capture | 33ms | 30fps |
| YOLO inference (OAK-D) | 30-50ms | Runs on Myriad X VPU |
| Fusion computation | 2-5ms | Simple scoring |
| **Total (parallelized)** | **~70-100ms** | Audio/video parallel |

**Jetson Nano** handles this pipeline comfortably since OAK-D offloads neural network inference. ODAS runs efficiently on ARM—originally designed for Raspberry Pi 3.

---

## Open-source projects to build from

These repositories provide tested, hackathon-ready components:

- **ODAS** (github.com/introlab/odas): Production-quality DOA with ROS integration via `odas_ros`
- **respeaker/usb_4_mic_array**: Python scripts for direct ReSpeaker DOA access
- **depthai-ros** (github.com/luxonis/depthai-ros): Official OAK-D ROS2 driver with spatial detection
- **pyroomacoustics**: DOA algorithm implementations (SRP-PHAT, MUSIC) for custom processing
- **Silero VAD**: State-of-the-art voice activity detection

For Johnny Five, the fastest path combines **ReSpeaker's built-in DOA** (via `tuning.py`) with **depthai-ros person detection**, connected through the fusion node template above. This avoids building ODAS from source while still achieving reliable speaker identification.

---

## Conclusion

The audio-visual fusion pipeline for Johnny Five requires three core components working together: a 4-mic array providing DOA angles (ReSpeaker + SRP-PHAT recommended), OAK-D running YOLO person detection, and a fusion node that maps DOA to pixel coordinates and scores detections probabilistically.

**Key implementation decisions**:
- Use **SRP-PHAT** for DOA—most robust in reverberant hackathon venues
- Map angles to pixels via `x = cx + f * tan(θ)` with OAK-D's known focal length
- Score candidates with **Gaussian likelihood** based on angular distance from DOA
- Apply **temporal consistency** (favor previous speaker) to prevent flickering
- Keep latency under 100ms by parallelizing audio and video pipelines

With proper calibration (mic array aligned with camera forward axis), the system reliably identifies speakers separated by more than 20°—sufficient for typical social robot interactions where people naturally space themselves apart when conversing.