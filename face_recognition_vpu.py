#!/usr/bin/env python3
"""VPU Face Recognition - All heavy lifting on OAK-D Myriad X

Uses three neural networks on VPU:
1. face-detection-retail-0004 - detect faces
2. head-pose-estimation-adas-0001 - head pose for alignment
3. face-recognition-arcface-112x112 - 512-d embeddings

Only cosine distance matching runs on Jetson CPU (trivial).
Based on: https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-recognition
"""

import os
import numpy as np
import depthai as dai
import blobconverter
import cv2
from pathlib import Path
import logging
from math import cos, sin

logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.expanduser("~/whoami/vpu_faces")

# Model paths (downloaded via blobconverter)
FACE_DET_MODEL = "face-detection-retail-0004"
HEAD_POSE_MODEL = "head-pose-estimation-adas-0001"
FACE_EMBED_MODEL = "face-reidentification-retail-0095"  # 256-d embeddings, 128x128 input


class VPUFaceRecognition:
    """Face recognition using OAK-D VPU for all NN inference"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.labels = []
        self.db_dic = {}
        self.admin = None
        self.load_database()

        self.pipeline = None
        self.device = None
        self.queues = {}

    def load_database(self):
        """Load face embeddings from npz files"""
        self.labels = []
        self.db_dic = {}

        # Load admin
        admin_file = self.db_path / "admin.txt"
        if admin_file.exists():
            self.admin = admin_file.read_text().strip()

        # Load embeddings
        for file in self.db_path.iterdir():
            if file.suffix == ".npz":
                name = file.stem
                self.labels.append(name)
                with np.load(file) as db:
                    self.db_dic[name] = [db[j] for j in db.files]

        logger.info(f"Loaded {len(self.labels)} faces: {self.labels}, Admin: {self.admin}")

    def save_embedding(self, name: str, embedding: np.ndarray):
        """Save a face embedding to database"""
        file_path = self.db_path / f"{name}.npz"

        # Load existing embeddings if any
        existing = []
        if file_path.exists():
            with np.load(file_path) as db:
                existing = [db[j] for j in db.files]

        existing.append(embedding)
        np.savez_compressed(file_path, *existing)

        # Set admin if first person
        if not self.admin:
            self.admin = name
            (self.db_path / "admin.txt").write_text(name)
            logger.info(f"Admin set to: {name}")

        # Reload database
        self.load_database()
        logger.info(f"Saved embedding for {name}")

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b.T) / (a_norm * b_norm))

    def recognize(self, embedding: np.ndarray) -> tuple:
        """Match embedding against database. Returns (name, confidence)"""
        if not self.labels:
            return "Unidentified", 0.0

        best_match = None
        best_conf = 0.0

        for label in self.labels:
            for stored_emb in self.db_dic.get(label, []):
                conf = self.cosine_distance(stored_emb, embedding)
                if conf > best_conf:
                    best_conf = conf
                    best_match = label

        # Threshold for recognition
        if best_conf >= 0.5:
            return best_match, best_conf
        return "Unidentified", best_conf

    def create_pipeline(self) -> dai.Pipeline:
        """Create DepthAI v3 pipeline with face detection on VPU

        Pipeline: Camera -> Face Detection -> Host
        Face cropping and embedding done on host for simplicity.
        Still saves massive memory vs running DeepFace on CPU!
        """
        logger.info("Creating VPU face detection pipeline (DepthAI v3)...")

        pipeline = dai.Pipeline()

        # Download face detection model
        face_det_blob = blobconverter.from_zoo(
            name=FACE_DET_MODEL,
            zoo_type="depthai",
            shaves=6
        )
        logger.info(f"Face detection model: {face_det_blob}")

        # === Camera (v3 API) ===
        cam = pipeline.create(dai.node.Camera).build()

        # NN input (300x300 for face-detection-retail-0004)
        nn_input = cam.requestOutput((300, 300), type=dai.ImgFrame.Type.BGR888p)

        # Display output (640x480)
        display_output = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)

        # === Face Detection (v3 API) ===
        face_det = pipeline.create(dai.node.DetectionNetwork)
        face_det.setBlobPath(face_det_blob)
        face_det.setConfidenceThreshold(0.5)
        nn_input.link(face_det.input)

        # Store outputs for queue creation
        self._display_output = display_output
        self._det_output = face_det.out

        logger.info("VPU face detection pipeline created")
        return pipeline

    def start(self):
        """Start the VPU pipeline (DepthAI v3 API)"""
        self.pipeline = self.create_pipeline()

        # Create output queues (v3 API - uses createOutputQueue on outputs)
        self.queues["color"] = self._display_output.createOutputQueue()
        self.queues["detections"] = self._det_output.createOutputQueue()

        # Start pipeline
        self.pipeline.start()

        logger.info("VPU face detection started")

    def stop(self):
        """Stop the pipeline"""
        if self.pipeline:
            self.pipeline.stop()
        self.pipeline = None
        self.queues = {}

    def get_frame_and_detections(self):
        """Get latest frame and face detections from VPU"""
        frame = None
        detections = []

        color_q = self.queues.get("color")
        det_q = self.queues.get("detections")

        if color_q:
            in_color = color_q.tryGet()
            if in_color:
                frame = in_color.getCvFrame()

        if det_q:
            in_det = det_q.tryGet()
            if in_det:
                detections = in_det.detections

        return frame, detections

    def crop_face(self, frame: np.ndarray, det) -> np.ndarray:
        """Crop face from frame using detection bbox"""
        h, w = frame.shape[:2]
        x1 = max(0, int(det.xmin * w))
        y1 = max(0, int(det.ymin * h))
        x2 = min(w, int(det.xmax * w))
        y2 = min(h, int(det.ymax * h))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def compute_embedding_cpu(self, face_crop: np.ndarray) -> np.ndarray:
        """Compute face embedding using DeepFace on CPU (lightweight)

        This is a fallback - uses the same database format.
        For full VPU embedding, need ImageManip pipeline.
        """
        try:
            from deepface import DeepFace
            # Use Facenet512 for 512-d embeddings compatible with our DB
            result = DeepFace.represent(
                face_crop,
                model_name="Facenet512",
                enforce_detection=False
            )
            if result and len(result) > 0:
                return np.array(result[0]["embedding"])
        except Exception as e:
            logger.debug(f"Embedding computation failed: {e}")
        return None


# Test VPU face detection
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    vpu = VPUFaceRecognition()
    vpu.start()

    frame_count = 0
    recognize_interval = 10  # Only compute embeddings every N frames per face

    try:
        while True:
            frame, detections = vpu.get_frame_and_detections()
            if frame is not None:
                frame_count += 1

                for det in detections:
                    h, w = frame.shape[:2]
                    x1 = int(det.xmin * w)
                    y1 = int(det.ymin * h)
                    x2 = int(det.xmax * w)
                    y2 = int(det.ymax * h)

                    # Crop face
                    face_crop = vpu.crop_face(frame, det)

                    # Recognize periodically (not every frame)
                    name = "Detecting..."
                    conf = det.confidence

                    if face_crop is not None and frame_count % recognize_interval == 0:
                        embedding = vpu.compute_embedding_cpu(face_crop)
                        if embedding is not None:
                            name, conf = vpu.recognize(embedding)

                    # Draw bbox
                    color = (0, 255, 0) if name not in ["Unidentified", "Detecting..."] else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{name} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow("VPU Face Detection", frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        vpu.stop()
        cv2.destroyAllWindows()
