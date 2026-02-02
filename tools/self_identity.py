#!/usr/bin/env python3
"""Self-Identity System

Chloe looks in a mirror and remembers herself.
Verbal command driven - human guides the process.

Flow:
1. Human: "Chloe, look in the mirror and remember yourself"
2. Chloe captures visual features from camera
3. Stores as "self" identity in MemoRable
4. Can now distinguish self from other robots

Usage:
    from tools.self_identity import get_self_identity

    identity = get_self_identity()

    # User says "remember yourself"
    result = await identity.look_in_mirror()

    # Later - check if a detection is "me"
    is_me = await identity.is_this_me(frame, bbox)
"""

import asyncio
import cv2
import numpy as np
import pickle
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime

# MemoRable for storing identity
try:
    from tools.memorable_client import get_memorable_client
    HAS_MEMORABLE = True
except:
    HAS_MEMORABLE = False


@dataclass
class SelfFeatures:
    """Visual features that identify 'me'."""
    # Color histogram of robot body
    color_histogram: Optional[np.ndarray] = None

    # Dominant colors
    dominant_colors: List[tuple] = field(default_factory=list)

    # Shape/contour signature
    contour_signature: Optional[np.ndarray] = None

    # Bounding box aspect ratio
    aspect_ratio: float = 0.0

    # Timestamp of capture
    captured_at: Optional[datetime] = None

    # Human-confirmed
    confirmed: bool = False


class SelfIdentity:
    """
    I know what I look like because I've seen myself.

    This is mirror-based self-recognition:
    - Robot looks in mirror
    - Human confirms "yes, that's you"
    - Robot stores visual signature
    - Can now recognize self vs other robots
    """

    def __init__(self):
        self.self_features: Optional[SelfFeatures] = None
        self.db_path = os.path.expanduser("~/whoami/self_identity.pkl")

        # Callbacks
        self._camera_callback: Optional[Callable[[], np.ndarray]] = None
        self._speak_callback: Optional[Callable[[str], None]] = None

        # Load existing self-knowledge
        self._load()

        # MemoRable client
        self._memorable = get_memorable_client() if HAS_MEMORABLE else None

    def set_camera_callback(self, callback: Callable[[], np.ndarray]):
        """Set callback to get camera frame."""
        self._camera_callback = callback

    def set_speak_callback(self, callback: Callable[[str], None]):
        """Set callback to speak to human."""
        self._speak_callback = callback

    def _speak(self, text: str):
        """Speak to the human."""
        if self._speak_callback:
            self._speak_callback(text)
        print(f"[SELF-IDENTITY]: {text}")

    def _load(self):
        """Load existing self-identity from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.self_features = data.get('features')
                    print(f"[SELF-IDENTITY]: Loaded self-knowledge from {self.db_path}")
            except Exception as e:
                print(f"[SELF-IDENTITY]: Failed to load: {e}")

    def _save(self):
        """Save self-identity to disk."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump({'features': self.self_features}, f)
            print(f"[SELF-IDENTITY]: Saved self-knowledge to {self.db_path}")
        except Exception as e:
            print(f"[SELF-IDENTITY]: Failed to save: {e}")

    async def look_in_mirror(self, roi: tuple = None) -> Dict[str, Any]:
        """
        Look in mirror and capture self-features.

        Called when human says "remember yourself" or "look in the mirror".

        Args:
            roi: Optional (x1, y1, x2, y2) region of interest where self is visible

        Returns:
            Result dict with success status and message
        """
        if not self._camera_callback:
            return {
                "success": False,
                "message": "I can't see - no camera connected",
            }

        self._speak("Looking in the mirror...")

        try:
            # Capture frame
            frame = self._camera_callback()
            if frame is None:
                return {
                    "success": False,
                    "message": "Camera returned no image",
                }

            # If ROI provided, use it; otherwise use center of frame
            if roi:
                x1, y1, x2, y2 = roi
            else:
                # Use center 60% of frame as likely self-location
                h, w = frame.shape[:2]
                x1 = int(w * 0.2)
                x2 = int(w * 0.8)
                y1 = int(h * 0.2)
                y2 = int(h * 0.8)

            # Extract region
            self_region = frame[y1:y2, x1:x2]

            # Compute features
            features = self._extract_features(self_region)
            features.captured_at = datetime.now()

            # Store temporarily (await confirmation)
            self._pending_features = features

            self._speak(
                "I can see something in the mirror. "
                "Is that me? Say 'yes' to confirm."
            )

            return {
                "success": True,
                "message": "Captured image, awaiting confirmation",
                "awaiting_confirmation": True,
                "dominant_colors": features.dominant_colors,
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error looking in mirror: {e}",
            }

    def _extract_features(self, region: np.ndarray) -> SelfFeatures:
        """Extract visual features from a region."""
        features = SelfFeatures()

        # Color histogram in HSV space
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        features.color_histogram = hist

        # Dominant colors via k-means
        pixels = region.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # Sort by frequency
            unique, counts = np.unique(labels, return_counts=True)
            sorted_idx = np.argsort(-counts)
            features.dominant_colors = [tuple(centers[i].astype(int)) for i in sorted_idx[:5]]
        except:
            pass

        # Aspect ratio
        h, w = region.shape[:2]
        features.aspect_ratio = w / h if h > 0 else 1.0

        # Contour signature (simplified edge histogram)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = np.histogram(edges.flatten(), bins=16, range=(0, 256))[0]
        features.contour_signature = edge_hist / (edge_hist.sum() + 1e-6)

        return features

    async def confirm_self(self) -> Dict[str, Any]:
        """
        Human confirmed the mirror image is self.

        Called when human says "yes" after look_in_mirror.
        """
        if not hasattr(self, '_pending_features') or self._pending_features is None:
            return {
                "success": False,
                "message": "Nothing to confirm - please look in mirror first",
            }

        self._pending_features.confirmed = True
        self.self_features = self._pending_features
        self._pending_features = None

        # Save locally
        self._save()

        # Store in MemoRable
        if self._memorable:
            try:
                await self._memorable.store(
                    content="I looked in the mirror and learned what I look like",
                    context={
                        "type": "self_identity",
                        "dominant_colors": [list(c) for c in self.self_features.dominant_colors],
                        "aspect_ratio": self.self_features.aspect_ratio,
                        "confirmed": True,
                    }
                )
            except Exception as e:
                print(f"[SELF-IDENTITY]: MemoRable store failed: {e}")

        self._speak("I'll remember what I look like. Thank you for showing me.")

        return {
            "success": True,
            "message": "Self-identity stored",
            "dominant_colors": self.self_features.dominant_colors,
        }

    async def reject_self(self) -> Dict[str, Any]:
        """
        Human rejected the mirror image - that's not me.

        Called when human says "no" after look_in_mirror.
        """
        self._pending_features = None
        self._speak("Oh, that wasn't me. Let's try again when you're ready.")

        return {
            "success": True,
            "message": "Rejected, ready to try again",
        }

    async def is_this_me(self, frame: np.ndarray, bbox: tuple) -> tuple:
        """
        Check if a detection is myself.

        Args:
            frame: Full camera frame
            bbox: (x1, y1, x2, y2) of detected object

        Returns:
            (is_self, confidence)
        """
        if self.self_features is None:
            return False, 0.0

        x1, y1, x2, y2 = bbox
        region = frame[y1:y2, x1:x2]

        if region.size == 0:
            return False, 0.0

        # Extract features from detection
        test_features = self._extract_features(region)

        # Compare color histogram
        hist_score = 0.0
        if self.self_features.color_histogram is not None and test_features.color_histogram is not None:
            hist_score = cv2.compareHist(
                self.self_features.color_histogram,
                test_features.color_histogram,
                cv2.HISTCMP_CORREL
            )

        # Compare contour signature
        contour_score = 0.0
        if self.self_features.contour_signature is not None and test_features.contour_signature is not None:
            contour_score = 1.0 - np.linalg.norm(
                self.self_features.contour_signature - test_features.contour_signature
            )

        # Compare aspect ratio
        aspect_score = 1.0 - abs(self.self_features.aspect_ratio - test_features.aspect_ratio)
        aspect_score = max(0, aspect_score)

        # Combined score
        confidence = (hist_score * 0.5 + contour_score * 0.3 + aspect_score * 0.2)
        confidence = max(0.0, min(1.0, confidence))

        # Threshold
        is_self = confidence > 0.6

        return is_self, confidence

    def knows_self(self) -> bool:
        """Check if we've learned what we look like."""
        return self.self_features is not None and self.self_features.confirmed

    def describe_self(self) -> str:
        """Get a verbal description of what we look like."""
        if not self.knows_self():
            return "I don't know what I look like yet. Can you show me a mirror?"

        colors = self.self_features.dominant_colors
        if not colors:
            return "I know what I look like, but I'm not sure how to describe it."

        # Convert BGR to color names (simplified)
        def color_name(bgr):
            b, g, r = bgr
            if r > 150 and g < 100 and b < 100:
                return "red"
            elif r < 100 and g > 150 and b < 100:
                return "green"
            elif r < 100 and g < 100 and b > 150:
                return "blue"
            elif r > 150 and g > 150 and b < 100:
                return "yellow"
            elif r > 150 and g > 150 and b > 150:
                return "white"
            elif r < 50 and g < 50 and b < 50:
                return "black"
            elif r > 100 and g > 100 and b > 100:
                return "gray"
            else:
                return "colorful"

        main_colors = [color_name(c) for c in colors[:3]]
        unique_colors = list(dict.fromkeys(main_colors))

        return f"I'm mostly {' and '.join(unique_colors[:2])}."


# Singleton
_self_identity: Optional[SelfIdentity] = None


def get_self_identity() -> SelfIdentity:
    """Get the self-identity instance."""
    global _self_identity
    if _self_identity is None:
        _self_identity = SelfIdentity()
    return _self_identity


# =============================================================================
# Tool Integration - For Hume EVI tool calls
# =============================================================================

async def handle_look_in_mirror(params: dict = None) -> dict:
    """Tool handler: Robot looks in mirror."""
    identity = get_self_identity()
    return await identity.look_in_mirror()


async def handle_confirm_self(params: dict = None) -> dict:
    """Tool handler: Confirm that's me in the mirror."""
    identity = get_self_identity()
    return await identity.confirm_self()


async def handle_reject_self(params: dict = None) -> dict:
    """Tool handler: Reject - that's not me."""
    identity = get_self_identity()
    return await identity.reject_self()


async def handle_describe_self(params: dict = None) -> dict:
    """Tool handler: Describe what I look like."""
    identity = get_self_identity()
    return {
        "success": True,
        "message": identity.describe_self(),
        "knows_self": identity.knows_self(),
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("Self-Identity Test")
        print("=" * 50)

        identity = get_self_identity()

        # Check if we know ourselves
        if identity.knows_self():
            print(f"I know myself: {identity.describe_self()}")
        else:
            print("I don't know what I look like yet.")

        # Simulate mirror look (would need real camera)
        print("\nTo test with camera:")
        print("1. Set camera callback")
        print("2. Call await identity.look_in_mirror()")
        print("3. Confirm with await identity.confirm_self()")

    asyncio.run(test())
