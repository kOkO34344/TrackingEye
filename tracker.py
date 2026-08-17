"""Hand tracking and gesture classification on top of MediaPipe Tasks."""

import os
from dataclasses import dataclass, field

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# Landmark indices we care about (MediaPipe hand model, 21 points).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

# Gestures
IDLE = "idle"
MOVE = "move"
PINCH = "pinch"
RIGHT_PINCH = "right-pinch"
SCROLL = "scroll"

# Skeleton edges, for drawing.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class HandFrame:
    """What one camera frame told us about the hand."""

    found: bool = False
    gesture: str = IDLE
    # Landmarks in mirrored normalized coords (x already flipped for selfie view).
    points: list = field(default_factory=list)
    # Stable anchor used for cursor movement.
    anchor: tuple = (0.0, 0.0)
    pinch_ratio: float = 1.0


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class HandTracker:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Missing model file: {MODEL_PATH}\n"
                "Re-download it with:\n"
                "  curl -L -o models/hand_landmarker.task https://storage.googleapis.com/"
                "mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        # Pinch state is sticky (hysteresis) so a borderline pinch doesn't chatter.
        self._index_pinched = False
        self._middle_pinched = False

    def close(self):
        self.landmarker.close()

    def process(self, rgb_frame, timestamp_ms):
        """rgb_frame is an un-mirrored RGB numpy array straight from the camera."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self._index_pinched = False
            self._middle_pinched = False
            return HandFrame(found=False)

        # Mirror x so moving your hand right moves the cursor right.
        points = [(1.0 - lm.x, lm.y) for lm in result.hand_landmarks[0]]
        gesture, pinch_ratio = self._classify(points)

        return HandFrame(
            found=True,
            gesture=gesture,
            points=points,
            # Index knuckle, not fingertip: it barely moves when you pinch,
            # so clicking doesn't drag the cursor off target.
            anchor=points[INDEX_MCP],
            pinch_ratio=pinch_ratio,
        )

    @staticmethod
    def measure(p):
        """Raw, threshold-free hand geometry. Used by classification and calibration.

        Distances are divided by hand size, so they mean the same thing whether
        your hand is close to the camera or far from it.
        """
        scale = max(_dist(p[WRIST], p[MIDDLE_MCP]), 1e-6)
        return {
            "index_ratio": _dist(p[THUMB_TIP], p[INDEX_TIP]) / scale,
            "middle_ratio": _dist(p[THUMB_TIP], p[MIDDLE_TIP]) / scale,
            # A finger counts as extended when its tip is above its middle joint.
            "index_up": p[INDEX_TIP][1] < p[INDEX_PIP][1],
            "middle_up": p[MIDDLE_TIP][1] < p[MIDDLE_PIP][1],
            "ring_up": p[RING_TIP][1] < p[RING_PIP][1],
            "pinky_up": p[PINKY_TIP][1] < p[PINKY_PIP][1],
        }

    def _classify(self, p):
        m = self.measure(p)
        index_ratio, middle_ratio = m["index_ratio"], m["middle_ratio"]
        index_up, middle_up = m["index_up"], m["middle_up"]
        ring_up, pinky_up = m["ring_up"], m["pinky_up"]

        self._index_pinched = _hysteresis(self._index_pinched, index_ratio)
        self._middle_pinched = _hysteresis(self._middle_pinched, middle_ratio)

        # A pinch needs the pinching finger extended, not just close to the thumb.
        # In a fist the thumb rests against the curled fingers, which is the same
        # distance as a pinch — without this the clutch gesture would fire clicks.
        index_pinch = self._index_pinched and index_up
        middle_pinch = self._middle_pinched and middle_up

        if index_pinch and middle_pinch:
            # Thumb is near both fingertips (they sit close together). Closer wins.
            if index_ratio <= middle_ratio:
                return PINCH, index_ratio
            return RIGHT_PINCH, middle_ratio
        if index_pinch:
            return PINCH, index_ratio
        if middle_pinch:
            return RIGHT_PINCH, middle_ratio
        if index_up and middle_up and not ring_up and not pinky_up:
            return SCROLL, index_ratio
        if index_up and not middle_up:
            return MOVE, index_ratio
        # Fist, open palm, anything else: clutch. Cursor holds still.
        return IDLE, index_ratio


def _hysteresis(currently_closed, ratio):
    if currently_closed:
        return ratio < config.PINCH_RELEASE
    return ratio < config.PINCH_CLOSE
