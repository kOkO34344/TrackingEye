"""Camera-free tests for the gesture classifier and the cursor state machine.

    .venv/bin/python test_gestures.py

Synthetic hand landmarks stand in for the camera, so this checks the logic
that is hard to eyeball live: fist vs pinch, click vs drag, clutch behaviour.
"""

import time

import config
from tracker import HandTracker, HandFrame, MOVE, PINCH, RIGHT_PINCH, SCROLL, IDLE
from main import Controller

failures = []


def check(label, got, expected):
    ok = got == expected
    if not ok:
        failures.append(label)
    print(f"{'ok  ' if ok else 'FAIL'} {label:38s} {got!r}" + ("" if ok else f" != {expected!r}"))


# --- synthetic hands ------------------------------------------------------
def hand(fingers, thumb_tip):
    """Upright hand. fingers: name -> extended?  thumb_tip: (x, y)."""
    p = [None] * 21
    p[0] = (0.50, 0.90)
    mcp_x = {"index": 0.45, "middle": 0.50, "ring": 0.55, "pinky": 0.60}
    joints = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
              "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    for name, (mcp, pip, dip, tip) in joints.items():
        x = mcp_x[name]
        p[mcp], p[pip] = (x, 0.60), (x, 0.50)
        p[dip], p[tip] = ((x, 0.45), (x, 0.40)) if fingers[name] else ((x, 0.52), (x, 0.56))
    p[1], p[2], p[3] = (0.40, 0.80), (0.36, 0.72), (0.34, 0.66)
    p[4] = thumb_tip
    return p


ALL_UP = {"index": True, "middle": True, "ring": True, "pinky": True}
POINT = {"index": True, "middle": False, "ring": False, "pinky": False}
TWO_UP = {"index": True, "middle": True, "ring": False, "pinky": False}
FIST = {"index": False, "middle": False, "ring": False, "pinky": False}
THUMB_OUT = (0.28, 0.66)


def classify(points):
    tracker = HandTracker()  # fresh instance: no sticky pinch state
    gesture, _ = tracker._classify(points)
    tracker.close()
    return gesture


print("gesture classification")
check("point -> move", classify(hand(POINT, THUMB_OUT)), MOVE)
check("thumb+index -> pinch", classify(hand(POINT, (0.455, 0.41))), PINCH)
check("thumb+middle -> right click", classify(hand(TWO_UP, (0.505, 0.41))), RIGHT_PINCH)
check("index+middle up -> scroll", classify(hand(TWO_UP, THUMB_OUT)), SCROLL)
check("open palm -> idle", classify(hand(ALL_UP, THUMB_OUT)), IDLE)
check("fist -> idle, not a pinch", classify(hand(FIST, (0.44, 0.62))), IDLE)

small = [(0.5 + (x - 0.5) * 0.35, 0.5 + (y - 0.5) * 0.35) for x, y in hand(POINT, (0.455, 0.41))]
check("distant (small) hand -> pinch", classify(small), PINCH)

tracker = HandTracker()
wobble = [tracker._classify(hand(POINT, (0.455, y)))[0] for y in (0.412, 0.402, 0.415, 0.404)]
tracker.close()
check("hysteresis: no chatter", len(set(wobble)), 1)


# --- cursor state machine -------------------------------------------------
class FakeMouse:
    def __init__(self):
        self.left_down = False
        self.log = []

    def move_by(self, dx, dy):
        self.log.append(("move", dx, dy))

    def left_press(self):
        if not self.left_down:
            self.left_down = True
            self.log.append(("press",))

    def left_release(self):
        if self.left_down:
            self.left_down = False
            self.log.append(("release",))

    def left_click(self):
        self.log.append(("click",))

    def right_click(self):
        self.log.append(("rclick",))

    def scroll(self, amount):
        self.log.append(("scroll", amount))

    def release_all(self):
        self.left_release()


def controller():
    mouse = FakeMouse()
    return mouse, Controller(mouse)


FRAME_DT = 1 / 30  # tests run faster than real time, so state dt explicitly


def feed(ctrl, gesture, x, y, frames=1, dt=FRAME_DT):
    for _ in range(frames):
        ctrl.update(HandFrame(found=True, gesture=gesture, points=[(x, y)] * 21, anchor=(x, y)),
                    dt=dt)


def kinds(mouse):
    return [event[0] for event in mouse.log]


print("\ncursor control")
mouse, ctrl = controller()
feed(ctrl, MOVE, 0.5, 0.5, 5)
mouse.log.clear()
feed(ctrl, MOVE, 0.6, 0.55, 6)
check("pointing moves the cursor", any(e[0] == "move" for e in mouse.log), True)

mouse, ctrl = controller()
feed(ctrl, PINCH, 0.5, 0.5, 3)
feed(ctrl, MOVE, 0.5, 0.5, 3)
check("quick pinch -> click", kinds(mouse), ["click"])

mouse, ctrl = controller()
feed(ctrl, PINCH, 0.5, 0.5, 3)
time.sleep(config.DRAG_HOLD_SECONDS + 0.05)
feed(ctrl, PINCH, 0.52, 0.5, 3)
held = mouse.left_down
feed(ctrl, MOVE, 0.52, 0.5, 3)
check("held pinch -> drag then drop", (held, mouse.left_down), (True, False))

mouse, ctrl = controller()
feed(ctrl, RIGHT_PINCH, 0.5, 0.5, 20)
check("right click fires once", kinds(mouse).count("rclick"), 1)

mouse, ctrl = controller()
feed(ctrl, SCROLL, 0.5, 0.5, 5)
mouse.log.clear()
feed(ctrl, SCROLL, 0.5, 0.7, 8)
scrolled = sum(e[1] for e in mouse.log if e[0] == "scroll")
check("hand down scrolls (natural)", scrolled > 0, config.SCROLL_NATURAL)

mouse, ctrl = controller()
feed(ctrl, MOVE, 0.5, 0.5, 5)
mouse.log.clear()
feed(ctrl, IDLE, 0.5, 0.5, 4)
feed(ctrl, IDLE, 0.1, 0.9, 4)      # reposition while clutched
feed(ctrl, MOVE, 0.1, 0.9, 5)      # re-engage somewhere else
check("clutch: re-engage doesn't fling", mouse.log, [])

mouse, ctrl = controller()
feed(ctrl, MOVE, 0.5, 0.5, 5)
mouse.log.clear()
feed(ctrl, MOVE, 0.9, 0.9, 1)      # tracking glitch, not a real movement
check("teleport frame dropped", mouse.log, [])

mouse, ctrl = controller()
feed(ctrl, MOVE, 0.5, 0.5, 5)
mouse.log.clear()
for step in range(1, 40):
    feed(ctrl, MOVE, 0.5 + step * 0.0008, 0.5)   # each step below the deadzone
check("slow drift still moves", sum(e[1] for e in mouse.log if e[0] == "move") > 20, True)

mouse, ctrl = controller()
feed(ctrl, PINCH, 0.5, 0.5, 3)
time.sleep(config.DRAG_HOLD_SECONDS + 0.05)
feed(ctrl, PINCH, 0.5, 0.5, 2)
ctrl.update(HandFrame(found=False))
check("hand lost mid-drag -> release", mouse.left_down, False)

mouse, ctrl = controller()
feed(ctrl, PINCH, 0.5, 0.5, 3)
time.sleep(config.DRAG_HOLD_SECONDS + 0.05)
feed(ctrl, PINCH, 0.5, 0.5, 2)
ctrl.set_armed(False)
mouse.log.clear()
feed(ctrl, MOVE, 0.7, 0.7, 6)
feed(ctrl, PINCH, 0.7, 0.7, 4)
check("disarmed emits nothing", mouse.log, [])


# --- pointer feel ---------------------------------------------------------
def travel(mouse):
    return sum(abs(e[1]) + abs(e[2]) for e in mouse.log if e[0] == "move")


print("\npointer feel")
# Same per-frame amplitude, once as jitter and once as deliberate motion.
# The adaptive filter should damp the first and pass the second.
mouse, ctrl = controller()
feed(ctrl, MOVE, 0.5, 0.5, 5)
mouse.log.clear()
for step in range(60):
    feed(ctrl, MOVE, 0.5 + (0.006 if step % 2 else -0.006), 0.5)
shaky = travel(mouse)

mouse, ctrl = controller()
feed(ctrl, MOVE, 0.5, 0.5, 5)
mouse.log.clear()
for step in range(60):
    feed(ctrl, MOVE, 0.5 + step * 0.006, 0.5)
deliberate = travel(mouse)

check("jitter damped vs real motion", deliberate > 3 * shaky, True)
check("deliberate motion gets through", deliberate > 0.5 * config.SENSITIVITY * 0.35, True)

# Acceleration: crossing the same hand distance quickly should cover more screen.
def sweep(frames, distance=0.3):
    mouse, ctrl = controller()
    feed(ctrl, MOVE, 0.5, 0.5, 5)
    mouse.log.clear()
    for step in range(1, frames + 1):
        feed(ctrl, MOVE, 0.5 + distance * step / frames, 0.5)
    feed(ctrl, MOVE, 0.5 + distance, 0.5, 12)   # let the filter settle
    return travel(mouse)


slow, fast = sweep(60), sweep(6)
check("fast sweep travels further", fast > 1.3 * slow, True)
check("acceleration stays capped", fast < config.ACCELERATION_MAX * slow * 1.15, True)

# --- calibration maths ----------------------------------------------------
import numpy as np
from calibrate import split_clusters, evaluate

print("\ncalibration")
rng = np.random.default_rng(1)
mixture = np.concatenate([rng.normal(0.25, 0.05, 60), rng.normal(1.10, 0.12, 60)])
closed, opened = split_clusters(mixture)
check("pinch/release clusters separated", (len(closed), len(opened)), (60, 60))
check("closed cluster is the low one", np.percentile(closed, 90) < 0.4, True)

steady = rng.normal(0.25, 0.05, 120)
closed, _ = split_clusters(steady)
check("steady hold stays one cluster", float(np.percentile(closed, 90)) < 0.4, True)


def recording(point, pinch_index, pinch_middle, right_middle, fist):
    """Build a calibration recording from per-pose sample generators."""
    return {
        "point": {"index": point, "middle": np.full(len(point), 1.4)},
        "pinch": {"index": pinch_index, "middle": pinch_middle},
        "middle": {"index": np.full(len(right_middle), 0.6), "middle": right_middle},
        "fist": {"index": fist, "middle": fist},
    }


def constant(value, n=120, spread=0.03):
    return rng.normal(value, spread, n)


clean = recording(point=constant(1.05), pinch_index=constant(0.22),
                  pinch_middle=constant(0.9), right_middle=constant(0.18),
                  fist=constant(0.14))
verdict = evaluate(clean)
check("clean recording accepted", verdict["status"], "ok")
check("threshold sits between the two",
      verdict["closed"] < verdict["pinch_close"] < verdict["open"], True)
check("release above close (hysteresis)", verdict["pinch_release"] > verdict["pinch_close"], True)

# The failure this run actually hit: pinch released for most of the recording.
half_held = np.concatenate([constant(0.22, 40), constant(0.68, 80)])
verdict = evaluate(recording(constant(1.05), half_held, constant(0.9),
                             constant(0.18), constant(0.14)))
check("unheld pinch rejected", verdict["status"], "not-held")

# The other one: the index-pinch step recorded a middle-finger pinch.
verdict = evaluate(recording(constant(1.05), constant(0.67), constant(0.13),
                             constant(0.15), constant(0.17)))
check("wrong finger detected", verdict["status"], "wrong-finger")

# Pinch and pointing hand too close to separate.
verdict = evaluate(recording(constant(0.40), constant(0.34), constant(0.9),
                             constant(0.18), constant(0.14)))
check("insufficient margin rejected", verdict["status"], "too-close")

# Never emit a razor-thin hysteresis band, whatever the input.
for gap_test in (0.13, 0.3, 0.8):
    v = evaluate(recording(constant(0.25 + gap_test), constant(0.20), constant(0.9),
                           constant(0.18), constant(0.14)))
    if v["status"] == "ok":
        check(f"usable hysteresis band (gap {gap_test})",
              v["pinch_release"] - v["pinch_close"] > 0.02, True)

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}"))
raise SystemExit(1 if failures else 0)
