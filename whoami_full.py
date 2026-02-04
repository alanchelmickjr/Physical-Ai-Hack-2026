#!/usr/bin/env python3
"""WhoAmI - Face recognition with Piper TTS speech and accessibility (child-like version)"""
import cv2
import depthai as dai
import numpy as np
import pickle
import os
import logging
import time
import subprocess
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IPC for communication with johnny5
try:
    from ipc import get_bus, Topic, VisionChannel
    HAS_IPC = True
except ImportError:
    HAS_IPC = False
    logger.warning("IPC not available, using file-based fallback")

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except:
    HAS_YOLO = False

try:
    import face_recognition
    HAS_FR = True
except:
    HAS_FR = False

SCREEN_W, SCREEN_H = 1920, 1080
PIPER_DIR = os.path.expanduser("~/piper")
PIPER_MODEL = os.path.expanduser("~/piper_voices/en_US-ryan-medium.onnx")
LAST_SEEN_FILE = os.path.expanduser("~/whoami/last_seen.pkl")
KEYBOARD_STATE_FILE = "/tmp/keyboard_state.txt"

class PiperTTS:
    """Text-to-speech using standalone Piper binary"""
    def __init__(self, piper_dir=PIPER_DIR, model_path=PIPER_MODEL):
        self.piper_dir = piper_dir
        self.model_path = model_path
        self.speaking = False
        self.current_text = ""
        self.text_display_until = 0

    def speak(self, text):
        """Speak text asynchronously"""
        if self.speaking:
            return  # Skip if already speaking
        self.current_text = text
        self.text_display_until = time.time() + 8  # Show text for 8 seconds
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text):
        """Actually speak (blocking) - with silence prefix for HDMI warmup"""
        self.speaking = True
        try:
            tmp_speech = '/tmp/johnny_speech.wav'
            tmp_final = '/tmp/johnny_final.wav'
            
            # Generate speech to file
            gen_cmd = f'cd {self.piper_dir} && echo "{text}" | LD_LIBRARY_PATH=. ./piper --model {self.model_path} --length-scale 1.1 --output_file {tmp_speech}'
            subprocess.run(gen_cmd, shell=True, timeout=15, stderr=subprocess.DEVNULL)
            
            # Add 0.5s silence prefix to let HDMI wake up
            pad_cmd = f'sox -n -r 22050 -c 1 /tmp/silence.wav trim 0.0 0.8 2>/dev/null; sox /tmp/silence.wav {tmp_speech} {tmp_final} 2>/dev/null'
            subprocess.run(pad_cmd, shell=True, timeout=5, stderr=subprocess.DEVNULL)
            
            # Play with PulseAudio
            play_cmd = f'paplay {tmp_final}'
            subprocess.run(play_cmd, shell=True, timeout=30, stderr=subprocess.DEVNULL)
            
            # Cleanup
            for f in [tmp_speech, tmp_final, '/tmp/silence.wav']:
                try:
                    os.unlink(f)
                except:
                    pass
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self.speaking = False

    def get_display_text(self):
        """Get text to display on screen (for accessibility)"""
        if time.time() < self.text_display_until:
            return self.current_text
        return ""


class LastSeenTracker:
    """Track when each person was last seen"""
    def __init__(self, filepath=LAST_SEEN_FILE):
        self.filepath = filepath
        self.last_seen = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'wb') as f:
                pickle.dump(self.last_seen, f)
        except Exception as e:
            logger.error(f"Save last_seen failed: {e}")

    def update(self, name):
        """Update last seen time for a person"""
        self.last_seen[name] = datetime.now()
        self._save()

    def get_time_since(self, name):
        """Get human-readable time since last seen"""
        if name not in self.last_seen:
            return None

        delta = datetime.now() - self.last_seen[name]
        secs = delta.total_seconds()

        if secs < 10:
            return "just now"
        elif secs < 60:
            return f"{int(secs)} seconds ago"
        elif secs < 3600:
            mins = int(secs / 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif secs < 86400:
            hours = int(secs / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(secs / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"


class TrackedPerson:
    """Tracked person with cached identity - avoids re-running face recognition"""
    def __init__(self, track_id):
        self.track_id = track_id
        self.current_name = "Unidentified"
        self.confidence = 0.0
        self.announced = False
        self.locked = False
        self.last_recognition_time = 0  # When we last ran face recognition
        self.recognition_attempts = 0   # How many times we've tried to identify
        self.last_seen_time = time.time()
        self.centroid = (0, 0)
        self.bbox = (0, 0, 0, 0)

    def update_position(self, bbox):
        """Update position without re-running face recognition"""
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.last_seen_time = time.time()

    def needs_recognition(self):
        """Check if we need to run face recognition on this person"""
        # Already locked to a known identity - only re-check every 30 seconds
        if self.locked and self.current_name != "Unidentified":
            return time.time() - self.last_recognition_time > 30.0
        # Still unidentified - try up to 5 times, then give up for 10 seconds
        if self.recognition_attempts >= 5:
            return time.time() - self.last_recognition_time > 10.0
        # Not locked yet - run recognition
        return True

    def set_identity(self, name, conf):
        """Set identity after face recognition"""
        self.last_recognition_time = time.time()
        self.recognition_attempts += 1
        if name != "Unidentified" and conf > 0.3:
            self.current_name = name
            self.confidence = conf
            self.locked = True

    def get_name(self):
        return self.current_name, self.confidence, self.locked

    def should_announce(self):
        """Return True if we should announce this person (first time locked)"""
        if self.locked and not self.announced:
            self.announced = True
            return True
        return False


class CentroidTracker:
    """Track people across frames using centroid distance"""
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}  # track_id -> TrackedPerson
        self.max_disappeared = max_disappeared  # Frames before removing
        self.disappeared = defaultdict(int)  # track_id -> frames missing

    def update(self, bboxes):
        """Update tracker with new bounding boxes. Returns list of TrackedPerson."""
        # No detections - mark all as disappeared
        if len(bboxes) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)
            return list(self.objects.values())

        # Calculate centroids for input bboxes
        input_centroids = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            input_centroids.append((cx, cy))

        # If no existing objects, register all
        if len(self.objects) == 0:
            for i, bbox in enumerate(bboxes):
                self._register(bbox)
            return list(self.objects.values())

        # Match existing objects to new detections
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].centroid for oid in object_ids]

        # Compute distance matrix
        D = np.zeros((len(object_centroids), len(input_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.sqrt((oc[0] - ic[0])**2 + (oc[1] - ic[1])**2)

        # Greedy matching - assign closest pairs
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            # Max distance threshold (300 pixels - allow for movement)
            if D[row, col] > 300:
                continue

            track_id = object_ids[row]
            self.objects[track_id].update_position(bboxes[col])
            self.disappeared[track_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (disappeared)
        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            track_id = object_ids[row]
            self.disappeared[track_id] += 1
            if self.disappeared[track_id] > self.max_disappeared:
                self._deregister(track_id)

        # Register new objects (unmatched detections)
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self._register(bboxes[col])

        return list(self.objects.values())

    def _register(self, bbox):
        person = TrackedPerson(self.next_id)
        person.update_position(bbox)
        self.objects[self.next_id] = person
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, track_id):
        del self.objects[track_id]
        del self.disappeared[track_id]


class WhoAmITouch:
    def __init__(self):
        self.running = True
        self.known_encodings = []
        self.known_names = []
        self.admin = None  # First enrolled face becomes admin
        self.db_path = os.path.expanduser('~/whoami/face_database.pkl')
        self.tracker = CentroidTracker(max_disappeared=60)  # ~16 seconds before losing track

        # TTS and last seen tracking
        self.tts = PiperTTS()
        self.last_seen = LastSeenTracker()

        # Speech history for display
        self.speech_history = deque(maxlen=3)

        # IPC for communication with johnny5
        self.vision_channel = None
        self._pending_enrollment = None
        if HAS_IPC:
            self.vision_channel = VisionChannel(get_bus(), source="whoami")
            # Subscribe to enrollment requests from voice system
            get_bus().subscribe(Topic.VISION_ENROLL_REQUEST, self._on_enroll_request)
            logger.info("IPC initialized")

        self.load_database()

        # YOLO - try VPU first, then CPU fallback
        self.yolo = None
        self.yolo_vpu = False  # Flag: True if using VPU, False if CPU
        self.nn_queue = None   # VPU detection queue

        self.pipeline = None
        self.device = None     # OAK-D device reference
        self.queue = None

    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                self.known_encodings = data.get('encodings', [])
                self.known_names = data.get('names', [])
                self.admin = data.get('admin', None)
                logger.info(f"Loaded DB: {self.known_names}, Admin: {self.admin}")
            except Exception as e:
                logger.error(f"DB load failed: {e}")

    def save_database(self):
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_encodings,
                    'names': self.known_names,
                    'admin': self.admin
                }, f)
            logger.info(f"Saved DB: {self.known_names}, Admin: {self.admin}")
        except Exception as e:
            logger.error(f"DB save failed: {e}")

    def enroll_face(self, name, encoding):
        """Enroll a new face. First enrolled becomes admin."""
        is_first = len(self.known_names) == 0
        self.known_encodings.append(encoding)
        self.known_names.append(name)
        if is_first or self.admin is None:
            self.admin = name
            logger.info(f"Admin set to: {name}")
        self.save_database()
        return is_first  # Returns True if this person is now admin

    def is_admin(self, name):
        """Check if a person is the admin."""
        return name == self.admin

    def start_camera(self):
        """Start camera with VPU YOLO if available, else CPU fallback"""
        # Try VPU YOLO first
        if self._try_vpu_yolo():
            return True

        # VPU failed, try camera-only with CPU YOLO
        logger.info("VPU unavailable, trying CPU YOLO fallback")
        return self._start_camera_cpu_yolo()

    def _try_vpu_yolo(self):
        """Try to start camera with YOLO on VPU using NeuralNetwork + DetectionParser"""
        try:
            # Find YOLO blob
            blob_paths = [
                os.path.expanduser("~/models/yolov8n_coco_640x352.blob"),
                os.path.expanduser("~/yolov8n_coco_640x352.blob"),
            ]
            blob_path = None
            for bp in blob_paths:
                if os.path.exists(bp):
                    blob_path = bp
                    break

            if not blob_path:
                raise Exception("No YOLO blob file found for VPU")

            # DepthAI 3.x Pipeline with NeuralNetwork for YOLO
            self.pipeline = dai.Pipeline()

            # Camera node
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(640, 480)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam_rgb.setFps(30)

            # NeuralNetwork for YOLO (not DetectionNetwork)
            nn = self.pipeline.create(dai.node.NeuralNetwork)
            nn.setBlobPath(blob_path)
            nn.setNumInferenceThreads(2)
            nn.input.setBlocking(False)

            # DetectionParser for YOLO output format
            parser = self.pipeline.create(dai.node.DetectionParser)
            parser.setNNFamily(dai.DetectionNetworkType.YOLO)
            parser.setConfidenceThreshold(0.5)
            parser.setNumClasses(80)  # COCO classes
            parser.setCoordinateSize(4)
            parser.setAnchors([])  # YOLOv8 is anchor-free
            parser.setAnchorMasks({})
            parser.setIouThreshold(0.5)

            # Link camera -> NN -> parser
            cam_rgb.preview.link(nn.input)
            nn.out.link(parser.input)

            # Output queues
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_rgb.preview.link(xout_rgb.input)

            xout_nn = self.pipeline.create(dai.node.XLinkOut)
            xout_nn.setStreamName("nn")
            parser.out.link(xout_nn.input)

            # Start pipeline
            self.device = dai.Device(self.pipeline)
            self.queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.nn_queue = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            self.yolo_vpu = True
            logger.info(f"VPU YOLO started with {blob_path}")
            return True

        except Exception as e:
            logger.warning(f"VPU YOLO failed: {e}")
            if hasattr(self, 'device') and self.device:
                try:
                    self.device.close()
                    self.device = None
                except:
                    pass
            self.pipeline = None
            return False

    def _start_camera_cpu_yolo(self):
        """Start camera-only, use CPU YOLO"""
        try:
            self.pipeline = dai.Pipeline()
            cam = self.pipeline.create(dai.node.Camera).build()
            output = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
            self.queue = output.createOutputQueue()
            self.pipeline.start()

            # Load CPU YOLO
            if HAS_YOLO:
                try:
                    self.yolo = YOLO('yolov8n.pt')
                    logger.info("CPU YOLO loaded")
                except Exception as e:
                    logger.error(f"CPU YOLO failed: {e}")

            self.yolo_vpu = False
            logger.info("Oak D started (CPU YOLO mode)")
            return True
        except Exception as e:
            logger.error(f"Camera failed: {e}")
            return False

    def recognize_face(self, frame, bbox):
        if not HAS_FR or not self.known_encodings:
            return "Unidentified", 0.0

        x1, y1, x2, y2 = bbox
        pad = 30
        y1p, y2p = max(0, y1 - pad), min(frame.shape[0], y2 + pad)
        x1p, x2p = max(0, x1 - pad), min(frame.shape[1], x2 + pad)

        face_img = frame[y1p:y2p, x1p:x2p]
        if face_img.size == 0:
            return "Unidentified", 0.0

        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        try:
            face_locations = face_recognition.face_locations(rgb_face, model='hog')
            if not face_locations:
                face_locations = [(0, face_img.shape[1], face_img.shape[0], 0)]

            encodings = face_recognition.face_encodings(rgb_face, face_locations)
            if not encodings:
                return "Unidentified", 0.0

            distances = face_recognition.face_distance(self.known_encodings, encodings[0])
            if len(distances) > 0:
                best_idx = np.argmin(distances)
                best_dist = distances[best_idx]
                logger.info(f"Distances: {dict(zip(self.known_names, [round(d,3) for d in distances]))}")
                if best_dist < 0.70:
                    return self.known_names[best_idx], 1.0 - best_dist
        except:
            pass

        return "Unidentified", 0.0

    def _on_enroll_request(self, msg):
        """Handle enrollment request from voice system via IPC."""
        name = msg.data.get("name", "")
        if name:
            self._pending_enrollment = name
            logger.info(f"Enrollment request received via IPC: {name}")

    def _send_greeting(self, text):
        """Send greeting via IPC or file fallback."""
        if self.vision_channel:
            self.vision_channel.publish_greeting(text)
        else:
            # File fallback for backward compatibility
            with open("/tmp/johnny5_greeting.txt", "w") as f:
                f.write(text)

    def announce_person(self, name):
        """Announce a person with their name and time since last seen - CHILD-LIKE version"""
        time_since = self.last_seen.get_time_since(name)

        if time_since is None:
            # First time seeing this person ever
            text = f"Hello {name}! Nice to meet you!"
        elif time_since == "just now":
            # Super recent - still announce but be playful
            text = f"Hi again {name}!"
        else:
            text = f"Hello {name}! I last saw you {time_since}."

        logger.info(f"Announcing: {text}")
        self.speech_history.append(text)
        self._send_greeting(text)
        self.last_seen.update(name)

    def prompt_enrollment(self):
        """Prompt unidentified person for their name. First person becomes admin."""
        if self.admin is None:
            text = "Hi there! I don't have an admin yet. Please tell me your name to become my admin."
        else:
            text = "Hi! I don't think we've met. What's your name?"
        logger.info(f"Enrollment prompt: {text}")
        self.speech_history.append(text)
        self._send_greeting(text)

    def draw_keyboard_icon(self, display):
        """Draw keyboard direction icon in top-right corner"""
        try:
            if not os.path.exists(KEYBOARD_STATE_FILE):
                return
            with open(KEYBOARD_STATE_FILE, 'r') as f:
                lines = f.read().strip().split('\n')
            if len(lines) < 2:
                return
            direction = lines[0]
            timestamp = float(lines[1])
            # Only show if recent (within 2 seconds)
            if time.time() - timestamp > 2:
                return
            if direction == "OFF":
                return

            # Draw direction indicator box
            box_x, box_y = SCREEN_W - 120, 60
            box_size = 100

            # Background
            cv2.rectangle(display, (box_x, box_y), (box_x + box_size, box_y + box_size), (40, 40, 40), -1)
            cv2.rectangle(display, (box_x, box_y), (box_x + box_size, box_y + box_size), (100, 100, 100), 2)

            cx, cy = box_x + box_size // 2, box_y + box_size // 2
            arrow_color = (0, 255, 0)  # Green

            # Draw arrow based on direction
            if direction == "FWD":
                pts = np.array([[cx, cy - 35], [cx - 25, cy + 20], [cx + 25, cy + 20]], np.int32)
                cv2.fillPoly(display, [pts], arrow_color)
            elif direction == "BACK":
                pts = np.array([[cx, cy + 35], [cx - 25, cy - 20], [cx + 25, cy - 20]], np.int32)
                cv2.fillPoly(display, [pts], arrow_color)
            elif direction == "LEFT":
                pts = np.array([[cx - 35, cy], [cx + 20, cy - 25], [cx + 20, cy + 25]], np.int32)
                cv2.fillPoly(display, [pts], arrow_color)
            elif direction == "RIGHT":
                pts = np.array([[cx + 35, cy], [cx - 20, cy - 25], [cx - 20, cy + 25]], np.int32)
                cv2.fillPoly(display, [pts], arrow_color)
            elif direction == "ROT_L":
                cv2.ellipse(display, (cx, cy), (30, 30), 0, 45, 315, arrow_color, 4)
                cv2.arrowedLine(display, (cx - 21, cy - 21), (cx - 30, cy - 10), arrow_color, 4)
            elif direction == "ROT_R":
                cv2.ellipse(display, (cx, cy), (30, 30), 0, 225, 495, arrow_color, 4)
                cv2.arrowedLine(display, (cx + 21, cy - 21), (cx + 30, cy - 10), arrow_color, 4)
            elif direction == "STOP":
                cv2.rectangle(display, (cx - 20, cy - 20), (cx + 20, cy + 20), (0, 0, 255), -1)
            elif direction == "MOVE":
                cv2.circle(display, (cx, cy), 25, arrow_color, -1)

        except:
            pass

    def draw_johnny5_status(self, display):
        """Draw Johnny5/Chloe voice status indicator"""
        try:
            # Check if voice system is running (johnny5.py or chloe_startup.py)
            johnny_running = os.path.exists("/tmp/johnny5.pid")
            if not johnny_running:
                # Fallback: check process list (cached, refresh every 5 sec)
                if not hasattr(self, '_johnny_check_time') or time.time() - self._johnny_check_time > 5:
                    # Check for either johnny5.py or chloe_startup.py
                    result1 = subprocess.run(['pgrep', '-f', 'johnny5.py'], capture_output=True)
                    result2 = subprocess.run(['pgrep', '-f', 'chloe_startup.py'], capture_output=True)
                    self._johnny_running_cache = (result1.returncode == 0 or result2.returncode == 0)
                    self._johnny_check_time = time.time()
                johnny_running = getattr(self, '_johnny_running_cache', False)

            # Draw status box
            box_x, box_y = SCREEN_W - 240, 60
            box_w, box_h = 110, 100

            # Background
            cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
            cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 100, 100), 2)

            # Status indicator
            cx, cy = box_x + box_w // 2, box_y + 45
            if johnny_running:
                # Green circle with "J5"
                cv2.circle(display, (cx, cy), 30, (0, 200, 0), -1)
                cv2.putText(display, "J5", (cx - 15, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Red circle with "X"
                cv2.circle(display, (cx, cy), 30, (0, 0, 200), -1)
                cv2.putText(display, "X", (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Label
            cv2.putText(display, "JOHNNY5", (box_x + 8, box_y + box_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        except:
            pass

    def draw_speech_text(self, display):
        """Draw speech text at bottom of screen for accessibility"""
        text = self.tts.get_display_text()
        if not text:
            return

        # Semi-transparent black background
        overlay = display.copy()
        cv2.rectangle(overlay, (0, SCREEN_H - 80), (SCREEN_W, SCREEN_H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Large white text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2

        # Get text size and center it
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # If text is too wide, reduce font size
        while text_w > SCREEN_W - 40 and font_scale > 0.6:
            font_scale -= 0.1
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        x = (SCREEN_W - text_w) // 2
        y = SCREEN_H - 30

        # White text with black outline for readability
        cv2.putText(display, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(display, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x > SCREEN_W - 160 and y > SCREEN_H - 60:
                # Restart the Jetson
                import subprocess
                subprocess.run('echo robot | sudo -S reboot', shell=True, check=False)

    def run(self):
        if not self.start_camera():
            return

        cv2.namedWindow('WhoAmI', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('WhoAmI', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('WhoAmI', self.mouse_callback)

        frame_count = 0
        detections = []

        # Startup announcement - disabled, EVI handles greetings
        # self.tts.speak("Johnny Five is alive! Face recognition ready.")

        while self.running:
            if self.queue and self.queue.has():
                frame = self.queue.get().getCvFrame()

                h, w = frame.shape[:2]
                scale = min(SCREEN_W / w, SCREEN_H / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))

                display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
                x_off = (SCREEN_W - new_w) // 2
                y_off = (SCREEN_H - new_h) // 2
                display[y_off:y_off+new_h, x_off:x_off+new_w] = resized

                # Run detection - VPU or CPU mode
                if frame_count % 8 == 0:
                    try:
                        bboxes = []  # Collect bounding boxes first

                        if self.yolo_vpu and self.nn_queue:
                            # VPU MODE - get detections from neural network queue
                            if self.nn_queue.has():
                                in_nn = self.nn_queue.get()
                                for detection in in_nn.detections:
                                    if detection.label != 0:  # Person class only
                                        continue
                                    x1 = int(detection.xmin * frame.shape[1])
                                    y1 = int(detection.ymin * frame.shape[0])
                                    x2 = int(detection.xmax * frame.shape[1])
                                    y2 = int(detection.ymax * frame.shape[0])
                                    bboxes.append((x1, y1, x2, y2))

                        elif self.yolo:
                            # CPU MODE - run ultralytics YOLO
                            results = self.yolo(frame, verbose=False, conf=0.5, classes=[0])
                            for r in results:
                                for box in r.boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                    bboxes.append((x1, y1, x2, y2))

                        # Update tracker with all bboxes - returns tracked persons
                        tracked_persons = self.tracker.update(bboxes)

                        # Only run face recognition on persons that need it
                        detections = []
                        for person in tracked_persons:
                            x1, y1, x2, y2 = person.bbox

                            # Only recognize if needed (new person or periodic recheck)
                            if person.needs_recognition():
                                face_h = int((y2 - y1) * 0.4)
                                face_bbox = (x1, y1, x2, y1 + face_h)
                                name, conf = self.recognize_face(frame, face_bbox)
                                person.set_identity(name, conf)
                                logger.info(f"Track {person.track_id}: recognized as {name} ({conf:.2f})")

                            # Check for announcement
                            if person.should_announce():
                                name, conf, locked = person.get_name()
                                if name != "Unidentified":
                                    self.announce_person(name)
                                else:
                                    self.prompt_enrollment()

                            name, conf, locked = person.get_name()
                            detections.append((x1, y1, x2, y2, name, conf, locked))

                    except Exception as e:
                        logger.error(f"Detection error: {e}")

                for det in detections:
                    x1, y1, x2, y2, name, conf, locked = det
                    dx1 = int(x1 * scale) + x_off
                    dy1 = int(y1 * scale) + y_off
                    dx2 = int(x2 * scale) + x_off
                    dy2 = int(y2 * scale) + y_off

                    color = (0, 255, 0) if name != "Unidentified" else (0, 255, 255)
                    thickness = 3 if locked else 2
                    cv2.rectangle(display, (dx1, dy1), (dx2, dy2), color, thickness)

                    label = name.upper()
                    if locked:
                        label = f"* {label}"

                    lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(display, (dx1, dy1-30), (dx1+lsz[0]+10, dy1), color, -1)
                    cv2.putText(display, label, (dx1+5, dy1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # Status bar - show VPU or CPU mode
                mode_str = "VPU" if self.yolo_vpu else "CPU"
                cv2.rectangle(display, (0, 0), (SCREEN_W, 45), (30, 30, 30), -1)
                cv2.putText(display, f"WhoAmI | {len(self.known_names)} faces | YOLO: {mode_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # RESTART button
                cv2.rectangle(display, (SCREEN_W-160, SCREEN_H-60), (SCREEN_W-10, SCREEN_H-10), (0, 0, 180), -1)
                cv2.putText(display, "RESTART", (SCREEN_W-145, SCREEN_H-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Draw accessibility text (speech displayed at bottom)
                self.draw_speech_text(display)

                # Draw keyboard direction icon
                self.draw_keyboard_icon(display)

                # Draw Johnny5 status indicator
                self.draw_johnny5_status(display)

                cv2.imshow('WhoAmI', display)
                frame_count += 1


            # Check for enrollment request (via IPC or file fallback)
            enroll_name = self._pending_enrollment
            self._pending_enrollment = None

            # File fallback if IPC not available
            if not enroll_name:
                enroll_file = "/tmp/johnny5_enroll.txt"
                if os.path.exists(enroll_file):
                    try:
                        with open(enroll_file, 'r') as f:
                            enroll_name = f.read().strip()
                        os.remove(enroll_file)
                    except:
                        pass

            if enroll_name and len(detections) > 0:
                try:
                    # Get encoding from current best detection
                    best_det = detections[0]
                    bbox = (int(best_det[0]), int(best_det[1]),
                            int(best_det[2]), int(best_det[3]))
                    x1, y1, x2, y2 = bbox
                    pad = 30
                    y1p, y2p = max(0, y1 - pad), min(frame.shape[0], y2 + pad)
                    x1p, x2p = max(0, x1 - pad), min(frame.shape[1], x2 + pad)
                    face_img = frame[y1p:y2p, x1p:x2p]
                    if face_img.size > 0:
                        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_locs = face_recognition.face_locations(rgb_face, model='hog')
                        if not face_locs:
                            face_locs = [(0, face_img.shape[1], face_img.shape[0], 0)]
                        encodings = face_recognition.face_encodings(rgb_face, face_locs)
                        if encodings:
                            is_first = self.enroll_face(enroll_name, encodings[0])
                            if is_first:
                                logger.info(f"Enrolled {enroll_name} as ADMIN")
                                self._send_greeting(f"Welcome {enroll_name}! You are now my admin.")
                                if self.vision_channel:
                                    self.vision_channel.publish_enroll_complete(enroll_name, is_admin=True)
                            else:
                                logger.info(f"Enrolled {enroll_name}")
                                self._send_greeting(f"Nice to meet you {enroll_name}! I'll remember you.")
                                if self.vision_channel:
                                    self.vision_channel.publish_enroll_complete(enroll_name, is_admin=False)
                except Exception as e:
                    logger.error(f"Enrollment failed: {e}")

            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

        try:
            if self.device:
                self.device.close()
            elif self.pipeline:
                self.pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = WhoAmITouch()
    app.run()
