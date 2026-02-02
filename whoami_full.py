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
    """Raw identification - no cooldown (YOLO on VPU is stable)"""
    def __init__(self):
        self.current_name = "Unidentified"
        self.confidence = 0.0
        self.announced = False

    def vote(self, name, conf):
        # Direct assignment - no voting needed with VPU YOLO
        if name != "Unidentified":
            self.current_name = name
            self.confidence = conf

    def get_name(self):
        is_locked = self.current_name != "Unidentified"
        return self.current_name, self.confidence, is_locked

    def should_announce(self):
        """Return True if we should announce this person (first time identified)"""
        if self.current_name != "Unidentified" and not self.announced:
            self.announced = True
            return True
        return False


class WhoAmITouch:
    def __init__(self):
        self.running = True
        self.known_encodings = []
        self.known_names = []
        self.db_path = os.path.expanduser('~/whoami/face_database.pkl')
        self.tracked = {}
        self.grid_size = 100

        # TTS and last seen tracking
        self.tts = PiperTTS()
        self.last_seen = LastSeenTracker()

        # Speech history for display
        self.speech_history = deque(maxlen=3)

        self.load_database()

        self.yolo = None
        if HAS_YOLO:
            try:
                self.yolo = YOLO('yolov8n.pt')
                logger.info("YOLO loaded")
            except Exception as e:
                logger.error(f"YOLO failed: {e}")

        self.pipeline = None
        self.queue = None

    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                self.known_encodings = data.get('encodings', [])
                self.known_names = data.get('names', [])
                logger.info(f"Loaded DB: {self.known_names}")
            except Exception as e:
                logger.error(f"DB load failed: {e}")

    def start_camera(self):
        try:
            self.pipeline = dai.Pipeline()
            cam = self.pipeline.create(dai.node.Camera).build()
            output = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
            self.queue = output.createOutputQueue()
            self.pipeline.start()
            logger.info("Oak D started")
            return True
        except Exception as e:
            logger.error(f"Camera failed: {e}")
            return False

    def get_grid_key(self, x1, y1, x2, y2):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return (cx // self.grid_size, cy // self.grid_size)

    def get_tracker(self, x1, y1, x2, y2):
        key = self.get_grid_key(x1, y1, x2, y2)
        if key not in self.tracked:
            self.tracked[key] = TrackedPerson()
        return self.tracked[key]

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
        # self.tts.speak(text)  # Disabled - using Johnny5
        with open("/tmp/johnny5_greeting.txt", "w") as f: f.write(text)
        self.last_seen.update(name)

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

                # Run YOLO every 5 frames instead of 3 to reduce lag
                if frame_count % 8 == 0 and self.yolo:
                    try:
                        results = self.yolo(frame, verbose=False, conf=0.5, classes=[0])
                        detections = []
                        for r in results:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                conf = float(box.conf[0])

                                face_h = int((y2 - y1) * 0.4)
                                face_bbox = (x1, y1, x2, y1 + face_h)

                                raw_name, raw_conf = self.recognize_face(frame, face_bbox)
                                tracker = self.get_tracker(x1, y1, x2, y2)
                                tracker.vote(raw_name, raw_conf)
                                name, match_conf, locked = tracker.get_name()

                                # Check if we should announce this person
                                if tracker.should_announce() and name != "Unidentified":
                                    self.announce_person(name)

                                detections.append((x1, y1, x2, y2, name, conf, locked))
                    except Exception as e:
                        logger.error(f"Error: {e}")

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

                # Status bar
                cv2.rectangle(display, (0, 0), (SCREEN_W, 45), (30, 30, 30), -1)
                cv2.putText(display, f"WhoAmI | {len(self.known_names)} faces", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # RESTART button
                cv2.rectangle(display, (SCREEN_W-160, SCREEN_H-60), (SCREEN_W-10, SCREEN_H-10), (0, 0, 180), -1)
                cv2.putText(display, "RESTART", (SCREEN_W-145, SCREEN_H-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Draw accessibility text (speech displayed at bottom)
                self.draw_speech_text(display)

                cv2.imshow('WhoAmI', display)
                frame_count += 1

            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = WhoAmITouch()
    app.run()
