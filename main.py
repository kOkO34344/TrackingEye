"""TrackingEye — control the macOS cursor with your hand.

  point (index finger up)      move the cursor, trackpad style
  pinch thumb + index          click  (hold it to drag)
  pinch thumb + middle         right click
  index + middle up            scroll by moving your hand up/down
  fist / open palm             clutch — cursor holds still, reposition freely

  SPACE   arm / disarm cursor control
  C       switch to the next camera
  Q, ESC  quit

Flags: --camera N, --pick-camera, --list-cameras, --stats
"""

import sys
import time

import cv2

import cameras
import config
from filters import OneEuro2D
from mouse import Mouse, can_post_events, request_post_event_access
from tracker import HandTracker, IDLE, MOVE, PINCH, RIGHT_PINCH, SCROLL, CONNECTIONS

GESTURE_COLORS = {
    IDLE: (150, 150, 150),
    MOVE: (80, 220, 255),
    PINCH: (90, 255, 120),
    RIGHT_PINCH: (255, 170, 80),
    SCROLL: (230, 130, 255),
}
GESTURE_LABELS = {
    IDLE: "idle (clutch)",
    MOVE: "move",
    PINCH: "left click / drag",
    RIGHT_PINCH: "right click",
    SCROLL: "scroll",
}


class Controller:
    """Turns a stream of HandFrames into cursor events."""

    def __init__(self, mouse):
        self.mouse = mouse
        self.armed = config.START_ARMED
        self.gesture = IDLE
        self._candidate = IDLE
        self._candidate_frames = 0
        self._smoothed = None
        self._prev = None
        self._raw_prev = None
        self._pinch_started_at = None
        self._filter = OneEuro2D(config.JITTER_CUTOFF, config.SPEED_COMPENSATION)
        self._last_update = None

    def set_armed(self, armed):
        self.armed = armed
        if not armed:
            self.reset()

    def reset(self):
        """Drop all motion state and let go of any held button."""
        self.mouse.release_all()
        self._pinch_started_at = None
        self._resync(None)

    def update(self, hand, dt=None):
        if dt is None:
            now = time.monotonic()
            dt = now - self._last_update if self._last_update else 1 / 30
            self._last_update = now

        if not hand.found:
            self._end_pinch(cancel=True)
            self._settle(IDLE)
            self._resync(None)
            return

        self._settle(hand.gesture)
        moving = self.gesture in (MOVE, PINCH, SCROLL)

        # Resync on a clutch, or when the raw landmark teleports (the hand was
        # re-detected elsewhere). Either way: snap to the hand, emit nothing.
        if not moving or self._teleported(hand.anchor):
            self._resync(hand.anchor)
        else:
            anchor = self._smooth(hand.anchor, dt)
            self._raw_prev = hand.anchor
            if self.gesture == SCROLL:
                self._scroll(anchor, dt)
            else:
                self._move(anchor, dt)

        if self.gesture == PINCH:
            self._continue_pinch()
        else:
            self._end_pinch(cancel=False)

    # --- gesture debounce -------------------------------------------------
    def _settle(self, gesture):
        if gesture == self.gesture:
            self._candidate_frames = 0
            return
        if gesture == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = gesture
            self._candidate_frames = 1

        if self._candidate_frames >= config.GESTURE_STABLE_FRAMES:
            previous = self.gesture
            self.gesture = gesture
            self._candidate_frames = 0
            self._on_enter(previous, gesture)

    def _on_enter(self, previous, gesture):
        if gesture == PINCH:
            self._pinch_started_at = time.monotonic()
        if gesture == RIGHT_PINCH and self.armed:
            self.mouse.right_click()
        # Switching between move and scroll shouldn't carry a stale delta.
        if (previous == SCROLL) != (gesture == SCROLL):
            self._prev = None

    # --- motion -----------------------------------------------------------
    def _teleported(self, anchor):
        if self._raw_prev is None:
            return True
        return max(abs(anchor[0] - self._raw_prev[0]),
                   abs(anchor[1] - self._raw_prev[1])) > config.MAX_JUMP

    def _resync(self, anchor):
        """Line the filters up with the hand without moving the cursor."""
        self._smoothed = anchor
        self._raw_prev = anchor
        self._prev = None
        self._filter.reset(anchor)

    def _smooth(self, anchor, dt):
        self._smoothed = self._filter(anchor, dt)
        return self._smoothed

    def _delta(self, anchor):
        if self._prev is None:
            self._prev = anchor
            return None
        dx = anchor[0] - self._prev[0]
        dy = anchor[1] - self._prev[1]
        if (dx * dx + dy * dy) ** 0.5 < config.DEADZONE:
            return None  # hold _prev so slow, deliberate movement still accumulates
        self._prev = anchor
        return dx, dy

    @staticmethod
    def _gain(dx, dy, dt):
        """Pointer acceleration: the faster the hand, the more screen per unit."""
        if config.ACCELERATION <= 0:
            return config.SENSITIVITY
        speed = (dx * dx + dy * dy) ** 0.5 / max(dt, 1e-6)
        boost = 1.0 + config.ACCELERATION * (speed / config.ACCELERATION_KNEE)
        return config.SENSITIVITY * min(boost, config.ACCELERATION_MAX)

    def _move(self, anchor, dt):
        delta = self._delta(anchor)
        if delta is None or not self.armed:
            return
        gain = self._gain(delta[0], delta[1], dt)
        self.mouse.move_by(delta[0] * gain, delta[1] * gain)

    def _scroll(self, anchor, dt):
        delta = self._delta(anchor)
        if delta is None or not self.armed:
            return
        amount = delta[1] * config.SCROLL_SENSITIVITY
        self.mouse.scroll(amount if config.SCROLL_NATURAL else -amount)

    # --- click vs drag ----------------------------------------------------
    def _continue_pinch(self):
        if not self.armed or self._pinch_started_at is None:
            return
        held = time.monotonic() - self._pinch_started_at
        if held >= config.DRAG_HOLD_SECONDS:
            self.mouse.left_press()  # no-op once already down

    def _end_pinch(self, cancel):
        if self._pinch_started_at is None:
            return
        held = time.monotonic() - self._pinch_started_at
        self._pinch_started_at = None
        if not self.armed:
            return
        if self.mouse.left_down:
            self.mouse.left_release()
        elif not cancel and held < config.DRAG_HOLD_SECONDS:
            self.mouse.left_click()


def draw_overlay(frame, hand, controller, fps, camera_label=""):
    height, width = frame.shape[:2]
    color = GESTURE_COLORS[controller.gesture]

    if hand.found:
        for start, end in CONNECTIONS:
            p1 = (int(hand.points[start][0] * width), int(hand.points[start][1] * height))
            p2 = (int(hand.points[end][0] * width), int(hand.points[end][1] * height))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        for x_norm, y_norm in hand.points:
            cv2.circle(frame, (int(x_norm * width), int(y_norm * height)), 4, color, -1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (width, 74), (24, 24, 24), -1)
    label = GESTURE_LABELS[controller.gesture] if hand.found else "no hand"
    cv2.putText(frame, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color if hand.found else (120, 120, 120), 2, cv2.LINE_AA)

    state = "ARMED" if controller.armed else "DISARMED"
    state_color = (90, 255, 120) if controller.armed else (90, 90, 255)
    cv2.putText(frame, state, (width - 190, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                state_color, 2, cv2.LINE_AA)
    hint = f"{fps:4.1f} fps   space: arm/disarm   c: camera   q: quit"
    cv2.putText(frame, hint, (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    if camera_label:
        cv2.putText(frame, camera_label, (width - 190, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

    if controller.mouse.left_down:
        cv2.putText(frame, "DRAGGING", (width // 2 - 70, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 255, 120), 2, cv2.LINE_AA)
    return frame


def open_camera(index, wait_seconds=30):
    """Open the camera, waiting out the macOS permission prompt.

    macOS asks for camera access asynchronously, so the first attempt fails
    while the dialog is still on screen. Keep retrying until it's answered.
    """
    deadline = time.monotonic() + wait_seconds
    announced = False
    while True:
        camera = cv2.VideoCapture(index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if camera.isOpened() and camera.read()[0]:
            return camera
        camera.release()
        if time.monotonic() >= deadline:
            return None
        if not announced:
            print(f"Waiting for camera {index}... if macOS is asking for "
                  "Camera permission, click Allow.")
            announced = True
        time.sleep(1.0)


def chosen_camera():
    """--camera N wins, then --pick-camera, then config, then the built-in camera."""
    default = config.CAMERA_INDEX if config.CAMERA_INDEX is not None else cameras.builtin_index()
    if "--camera" in sys.argv:
        return int(sys.argv[sys.argv.index("--camera") + 1])
    if "--pick-camera" in sys.argv:
        return cameras.prompt_for_camera(default)
    return default


def main():
    if "--list-cameras" in sys.argv:
        print(cameras.describe())
        return

    index = chosen_camera()
    print(f"camera {index}: {cameras.camera_name(index)}")
    camera = open_camera(index)
    if camera is None:
        sys.exit(
            f"Could not open camera {index}. Enable Camera for your terminal app in\n"
            "System Settings > Privacy & Security > Camera, then restart that terminal\n"
            "(the permission only applies to new processes). Available cameras:\n"
            + cameras.describe()
        )

    tracker = HandTracker()
    mouse = Mouse()
    controller = Controller(mouse)

    # Without Accessibility the cursor events go nowhere and the app looks broken
    # while the tracking is actually fine. Say so loudly, and ask for it once.
    has_accessibility = can_post_events()
    if not has_accessibility:
        request_post_event_access()
        print("\n!! No Accessibility permission — tracking will run, but the cursor "
              "will not move.\n   System Settings > Privacy & Security > Accessibility: "
              "enable your terminal app,\n   then restart it and run this again.\n")

    print(__doc__)
    print(f"screen: {int(mouse.width)}x{int(mouse.height)}   "
          f"starting {'ARMED' if controller.armed else 'DISARMED'}\n")

    fps = 0.0
    last = time.monotonic()
    start = last

    # --stats prints a line a second: frame rate, how often the hand was found,
    # and which gestures fired. Useful for judging tracking quality from a log.
    stats_on = "--stats" in sys.argv
    stats_since = last
    seen_frames = 0
    seen_hands = 0
    seen_gestures = {}

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Dropped frame from camera, retrying...")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            hand = tracker.process(rgb, int((now - start) * 1000))
            controller.update(hand, dt=min(now - last, 0.2))

            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            if stats_on:
                seen_frames += 1
                seen_hands += hand.found
                if hand.found:
                    seen_gestures[controller.gesture] = seen_gestures.get(controller.gesture, 0) + 1
                if now - stats_since >= 1.0:
                    breakdown = " ".join(f"{name}:{count}" for name, count
                                         in sorted(seen_gestures.items(), key=lambda kv: -kv[1]))
                    print(f"[stats] {seen_frames / (now - stats_since):4.1f} fps  "
                          f"hand {100 * seen_hands / max(seen_frames, 1):3.0f}%  "
                          f"{breakdown or '-'}", flush=True)
                    stats_since = now
                    seen_frames = seen_hands = 0
                    seen_gestures = {}

            view = draw_overlay(cv2.flip(frame, 1), hand, controller, fps,
                                cameras.camera_name(index))
            if not has_accessibility:
                cv2.putText(view, "no Accessibility permission - cursor won't move",
                            (16, view.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (90, 90, 255), 2, cv2.LINE_AA)
            cv2.imshow("TrackingEye", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                controller.set_armed(not controller.armed)
                print("ARMED" if controller.armed else "DISARMED")
            if key == ord("c"):
                candidate = cameras.next_index(index)
                if candidate != index:
                    swapped = open_camera(candidate, wait_seconds=5)
                    if swapped is None:
                        print(f"Could not open camera {candidate}, staying on {index}.")
                    else:
                        camera.release()
                        camera, index = swapped, candidate
                        controller.reset()
                        print(f"camera {index}: {cameras.camera_name(index)}")
    except KeyboardInterrupt:
        pass
    finally:
        mouse.release_all()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
