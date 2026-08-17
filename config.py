"""Tunable knobs for TrackingEye. Edit these, no code changes needed."""

# --- Camera ---
# None = auto-pick the laptop's built-in camera. Set an integer to force a
# specific device; `python cameras.py` lists them. Auto-pick matters because
# Continuity Camera can put your iPhone at index 0.
CAMERA_INDEX = None
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# --- Pointer feel (trackpad-style relative movement) ---
# Screen pixels moved per 1.0 of normalized hand movement across the frame.
# Higher = faster cursor. Start here and tune to taste.
SENSITIVITY = 3400.0

# Extra gain on fast sweeps, like macOS pointer acceleration: slow movement stays
# precise, quick movement crosses the screen. 0 = uniform speed everywhere.
ACCELERATION = 1.6
ACCELERATION_KNEE = 0.9   # hand speed (frame widths/sec) where the boost kicks in
ACCELERATION_MAX = 2.6    # never multiply speed by more than this

# Hand jitter below this (in normalized frame units) is ignored entirely.
DEADZONE = 0.0035

# One Euro filter. JITTER_CUTOFF governs how still the cursor sits when your hand
# is resting: lower = steadier, but too low feels syrupy. SPEED_COMPENSATION buys
# back responsiveness as you move: higher = less lag on fast movement.
JITTER_CUTOFF = 1.0
SPEED_COMPENSATION = 8.0

# Movement larger than this in one frame is treated as a tracking glitch
# (hand re-detected somewhere else) and dropped instead of flinging the cursor.
MAX_JUMP = 0.25

# --- Gestures ---
# Thumb-fingertip distance (normalized, scaled by hand size) below which a
# pinch counts as "closed".  Hysteresis: must exceed RELEASE to open again.
# Measured with `./run.sh --calibrate`: index pinch closes to 0.05, right-click
# pinch to 0.23, pointing hand sits at 0.62 and never came below 0.53.
PINCH_CLOSE = 0.35
PINCH_RELEASE = 0.46

# A pinch held longer than this becomes a drag instead of a click.
DRAG_HOLD_SECONDS = 0.35

# Frames a gesture must persist before it is accepted (debounce).
GESTURE_STABLE_FRAMES = 2

# Scroll pixels per 1.0 of normalized vertical hand movement.
SCROLL_SENSITIVITY = 900.0
SCROLL_NATURAL = True  # True = content follows hand, like macOS natural scrolling

# --- Detection ---
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# --- Startup ---
START_ARMED = True  # False = launches disarmed; press SPACE to take control
