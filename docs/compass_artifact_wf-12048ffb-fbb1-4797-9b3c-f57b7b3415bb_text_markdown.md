# Voice-based speaker ID for Johnny Five on Jetson Orin

Adding voice biometrics to your robot requires selecting the right embedding model, optimizing for edge deployment, and integrating with your existing face recognition pipeline. **TitaNet-Small or ECAPA-TDNN with TensorRT optimization delivers sub-15ms inference on Orin**, enabling real-time speaker verification alongside your WhoAmI face system. This guide provides a complete implementation path using WeSpeaker's production-ready ONNX models, Silero VAD for speech detection, and multi-modal fusion with your Gun.js identity store.

## Embedding model selection drives everything else

Your choice of speaker embedding model determines accuracy, memory footprint, and optimization complexity. After evaluating four leading options, **WeSpeaker's ECAPA-TDNN** emerges as the best choice for Johnny Five—it offers native ONNX export, **0.72% EER on VoxCeleb**, and a compact 192-dimensional embedding that pairs efficiently with face vectors.

| Model | Embedding Dim | Size | EER (VoxCeleb1-O) | ONNX Support | Recommendation |
|-------|--------------|------|-------------------|--------------|----------------|
| **WeSpeaker ECAPA-TDNN** | 192 | ~60 MB | 0.72% | ✅ Native | **Best for Johnny Five** |
| TitaNet-Small (NeMo) | 192 | ~25 MB | 0.92% | ✅ Via export.py | Runner-up (NVIDIA native) |
| Resemblyzer | 256 | ~17 MB | 4.5% | ✅ TorchScript | Fastest prototyping |
| pyannote embedding | 512 | ~24 MB | 2.8% | ⚠️ Limited | Best for diarization |

WeSpeaker provides pre-exported ONNX models on HuggingFace (`Wespeaker/wespeaker-ecapa-tdnn512-LM`), eliminating export headaches. The 192-d embedding aligns with TitaNet's dimension, allowing easy model swapping later. For the 8GB Orin, the complete voice pipeline (model + VAD + buffers) consumes approximately **200-300 MB** in FP16, leaving ample headroom for your face recognition stack.

## TensorRT conversion unlocks real-time performance

Raw PyTorch inference on Jetson is sluggish. Converting to TensorRT delivers **4-8x speedup** with FP16 quantization adding minimal accuracy loss (~0.1-0.5% EER degradation). Here's the optimized deployment pipeline:

```bash
# Convert WeSpeaker ONNX to TensorRT on your Jetson Orin
/usr/src/tensorrt/bin/trtexec \
    --onnx=wespeaker_ecapa_tdnn.onnx \
    --saveEngine=speaker_model.trt \
    --fp16 \
    --workspace=2048 \
    --minShapes=audio_features:1x100x80 \
    --optShapes=audio_features:1x300x80 \
    --maxShapes=audio_features:1x500x80
```

The dynamic shape profiles handle variable-length utterances (1-5 seconds). For enrollment, use the optimal 300-frame input (~3 seconds); for verification, accept shorter utterances down to 100 frames (~1 second). Build engines directly on your target Orin—TensorRT engines aren't portable across GPU architectures.

```python
# TensorRT inference wrapper for Johnny Five
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class SpeakerEmbedder:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
    def extract(self, mel_features: np.ndarray) -> np.ndarray:
        """Extract 192-d speaker embedding from mel spectrogram."""
        # Allocate buffers
        input_buf = cuda.mem_alloc(mel_features.nbytes)
        output_buf = cuda.mem_alloc(192 * 4)  # 192-d float32
        
        cuda.memcpy_htod_async(input_buf, mel_features, self.stream)
        self.context.execute_async_v2([int(input_buf), int(output_buf)], self.stream.handle)
        
        embedding = np.empty(192, dtype=np.float32)
        cuda.memcpy_dtoh_async(embedding, output_buf, self.stream)
        self.stream.synchronize()
        return embedding / np.linalg.norm(embedding)  # L2 normalize
```

Memory budget for multi-model deployment looks healthy: speaker model (~200 MB) + ArcFace face model (~300 MB) + preprocessing buffers (~200 MB) = **~700 MB**, well under your 8GB ceiling with ~5GB available for other processes.

## Real-time verification needs Silero VAD and smart thresholds

Continuous speaker verification requires detecting when someone is actually speaking before extracting embeddings. **Silero VAD** is ideal for Jetson—MIT licensed, <1ms per 30ms chunk, and available as a compact ONNX model:

```python
from silero_vad import load_silero_vad, get_speech_timestamps

class VoiceActivityPipeline:
    def __init__(self, embedder: SpeakerEmbedder):
        self.vad = load_silero_vad(onnx=True)  # ~2MB ONNX model
        self.embedder = embedder
        self.speech_buffer = []
        self.MIN_SPEECH_FRAMES = 150  # ~1.5 seconds
        
    def process_audio_chunk(self, audio_16khz: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Process 30ms audio chunk, return (has_speech, embedding_if_ready)."""
        timestamps = get_speech_timestamps(audio_16khz, self.vad, 
            sampling_rate=16000, threshold=0.5, min_speech_duration_ms=250)
        
        if timestamps:
            self.speech_buffer.extend(audio_16khz)
            
            if len(self.speech_buffer) >= self.MIN_SPEECH_FRAMES * 160:  # frames * samples_per_frame
                mel = self._extract_mel(np.array(self.speech_buffer))
                embedding = self.embedder.extract(mel)
                self.speech_buffer = self.speech_buffer[-8000:]  # Keep 0.5s overlap
                return True, embedding
        return bool(timestamps), None
```

For cosine similarity thresholds, research shows **0.5-0.6 provides balanced FAR/FRR** for text-independent verification. Johnny Five should use adaptive thresholds based on context:

| Scenario | Threshold | Rationale |
|----------|-----------|-----------|
| Initial greeting | 0.55 | Balance convenience vs. security |
| After face match | 0.45 | Lower bar when face already verified |
| Voice-only (dark room) | 0.65 | Higher bar without visual confirmation |
| Continuous verification | 0.50 | Running average smooths outliers |

For **short utterance enrollment (2-5 seconds)**, accuracy degrades approximately 46% when reducing from 3.5s to 2s. Mitigate this by collecting 3-5 utterances and using L2-normalized averaging:

```python
def enroll_speaker(utterance_embeddings: list[np.ndarray]) -> np.ndarray:
    """Aggregate multiple enrollment embeddings into robust centroid."""
    normalized = [emb / np.linalg.norm(emb) for emb in utterance_embeddings]
    centroid = np.mean(normalized, axis=0)
    return centroid / np.linalg.norm(centroid)
```

**Incremental embedding updates** handle voice changes over time using exponential moving average:

```python
class AdaptiveEmbedding:
    def __init__(self, initial: np.ndarray, alpha: float = 0.1):
        self.centroid = initial
        self.alpha = alpha
        
    def update(self, new_embedding: np.ndarray, similarity: float) -> np.ndarray:
        # Only update if verification passed (similarity > threshold)
        if similarity > 0.55:
            effective_alpha = self.alpha * min(similarity, 1.0)
            self.centroid = (1 - effective_alpha) * self.centroid + effective_alpha * new_embedding
            self.centroid /= np.linalg.norm(self.centroid)
        return self.centroid
```

## Multi-modal storage schema for Gun.js integration

Gun.js doesn't natively support arrays, so embeddings must be Base64-encoded. Here's a schema that integrates voice embeddings with your existing WhoAmI face data:

```javascript
// Gun.js identity schema for Johnny Five
const personSchema = {
  personId: 'uuid-v4-string',
  
  face: {
    embedding: 'base64-encoded-float32-array',  // 512-d from WhoAmI → ~2.7KB
    model: 'arcface-r100',
    confidence: 0.95,
    enrolledAt: 1706745600000,
    version: 1
  },
  
  voice: {
    embedding: 'base64-encoded-float32-array',  // 192-d → ~1KB  
    model: 'wespeaker-ecapa-tdnn',
    confidence: 0.88,
    netSpeechSeconds: 12.5,
    enrolledAt: 1706745700000,
    version: 1
  },
  
  fusion: {
    faceWeight: 0.6,
    voiceWeight: 0.4,
    strategy: 'weighted_score'
  },
  
  metadata: {
    displayName: 'Alice',
    lastSeen: 1706832000000,
    totalInteractions: 47
  }
};

// Base64 conversion utilities
function float32ToBase64(arr) {
  const bytes = new Uint8Array(arr.buffer);
  return btoa(String.fromCharCode(...bytes));
}

function base64ToFloat32(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
```

For **multi-modal verification**, fuse face and voice scores with weighted averaging:

```python
def verify_multimodal(face_emb: np.ndarray, voice_emb: np.ndarray, 
                      enrolled: dict, threshold: float = 0.6) -> dict:
    """Combined face + voice verification for Johnny Five."""
    face_score = np.dot(face_emb, base64_to_float32(enrolled['face']['embedding']))
    voice_score = np.dot(voice_emb, base64_to_float32(enrolled['voice']['embedding']))
    
    # Confidence-weighted fusion
    face_w = enrolled['fusion']['faceWeight'] * enrolled['face']['confidence']
    voice_w = enrolled['fusion']['voiceWeight'] * enrolled['voice']['confidence']
    
    fused_score = (face_score * face_w + voice_score * voice_w) / (face_w + voice_w)
    
    return {
        'verified': fused_score >= threshold,
        'score': float(fused_score),
        'face_score': float(face_score),
        'voice_score': float(voice_score),
        'modalities': ['face', 'voice']
    }
```

When one modality is unavailable (dark room, silent user), fall back to single-modality verification with a **10-15% higher threshold** to compensate for reduced confidence.

## Enrollment workflow during first conversation

Johnny Five should seamlessly enroll new users during natural interaction. Here's the conversational enrollment flow:

```python
class FirstConversationEnrollment:
    """Handles enrollment during Johnny Five's first meeting with someone."""
    
    def __init__(self, face_model, speaker_model, gun_storage):
        self.face = face_model
        self.speaker = speaker_model  
        self.storage = gun_storage
        self.MIN_VOICE_SECONDS = 10  # Accumulate during conversation
        self.MIN_FACE_SAMPLES = 3
        
    async def start_enrollment(self, person_id: str):
        return EnrollmentSession(person_id, self)
    
class EnrollmentSession:
    def __init__(self, person_id: str, parent: FirstConversationEnrollment):
        self.person_id = person_id
        self.parent = parent
        self.face_samples = []
        self.voice_buffer = []
        self.voice_seconds = 0.0
        
    def add_face_frame(self, frame) -> dict:
        """Called each video frame during conversation."""
        detection = self.parent.face.detect(frame)
        if detection and detection.quality > 0.7:
            self.face_samples.append(detection.embedding)
        return {'face_samples': len(self.face_samples), 
                'face_ready': len(self.face_samples) >= self.parent.MIN_FACE_SAMPLES}
    
    def add_voice_chunk(self, audio_chunk, speech_duration: float) -> dict:
        """Called when VAD detects speech."""
        self.voice_buffer.append(audio_chunk)
        self.voice_seconds += speech_duration
        return {'voice_seconds': self.voice_seconds,
                'voice_ready': self.voice_seconds >= self.parent.MIN_VOICE_SECONDS}
    
    async def finalize(self) -> dict:
        """Complete enrollment after sufficient samples collected."""
        if len(self.face_samples) < self.parent.MIN_FACE_SAMPLES:
            return {'error': 'insufficient_face_samples'}
        if self.voice_seconds < self.parent.MIN_VOICE_SECONDS:
            return {'error': 'insufficient_voice_samples'}
            
        # Aggregate embeddings
        face_embedding = np.mean(self.face_samples, axis=0)
        face_embedding /= np.linalg.norm(face_embedding)
        
        voice_audio = np.concatenate(self.voice_buffer)
        voice_embedding = self.parent.speaker.extract(voice_audio)
        
        # Store in Gun.js
        await self.parent.storage.put_person(self.person_id, {
            'face': {'embedding': float32_to_base64(face_embedding), 
                     'confidence': 0.9, 'version': 1},
            'voice': {'embedding': float32_to_base64(voice_embedding),
                      'confidence': min(self.voice_seconds / 20, 1.0), 'version': 1},
            'fusion': {'faceWeight': 0.6, 'voiceWeight': 0.4}
        })
        
        return {'enrolled': True, 'person_id': self.person_id}
```

## Production-ready repositories to build from

Rather than building from scratch, leverage these actively maintained projects:

**WeSpeaker** (`github.com/wenet-e2e/wespeaker`) provides the most production-ready speaker verification toolkit with native ONNX export and TensorRT support. Clone it, export the ECAPA-TDNN model to ONNX, and convert to TensorRT on your Orin.

**Silero VAD** (`github.com/snakers4/silero-vad`) delivers enterprise-grade voice activity detection under MIT license with explicit Jetson Orin support documented.

**jetson-voice** (`github.com/dusty-nv/jetson-voice`) from NVIDIA provides TensorRT-optimized audio pipelines specifically for Jetson, though it's slightly dated (JetPack 4.x focused).

For ROS integration, **ros2_whisper** (`github.com/ros-ai/ros2_whisper`) demonstrates the pattern for real-time audio processing in ROS 2 with CUDA support. You'll need to create a custom speaker verification node following its architecture:

```python
# Minimal ROS 2 speaker verification node structure
import rclpy
from rclpy.node import Node
from audio_msgs.msg import AudioData
from std_msgs.msg import String

class SpeakerVerificationNode(Node):
    def __init__(self):
        super().__init__('speaker_verification')
        self.audio_sub = self.create_subscription(AudioData, '/audio', self.audio_cb, 10)
        self.identity_pub = self.create_publisher(String, '/speaker_identity', 10)
        self.embedder = SpeakerEmbedder('speaker_model.trt')
        self.vad_pipeline = VoiceActivityPipeline(self.embedder)
        
    def audio_cb(self, msg):
        has_speech, embedding = self.vad_pipeline.process_audio_chunk(msg.data)
        if embedding is not None:
            identity = self.verify_against_enrolled(embedding)
            self.identity_pub.publish(String(data=identity))
```

## Quick-start implementation checklist

To add voice-based speaker identification to Johnny Five:

1. **Install WeSpeaker** and download the ECAPA-TDNN ONNX model from HuggingFace
2. **Convert to TensorRT** on your Jetson Orin using the `trtexec` command above
3. **Add Silero VAD** for speech detection (`pip install silero-vad`)
4. **Extend your Gun.js schema** to include `voice` field alongside existing face data
5. **Implement enrollment** by accumulating 10+ seconds of speech during first conversation
6. **Add multi-modal fusion** to your verification loop with weighted scoring
7. **Create a ROS 2 node** subscribing to audio topics and publishing speaker identity

Total memory footprint: ~700 MB for speaker + face models. Latency budget: ~50-100ms total pipeline (VAD + embedding extraction + similarity computation). With the 192-d embedding and cosine similarity threshold of 0.55, expect **>95% verification accuracy** for enrolled speakers in typical indoor conditions.