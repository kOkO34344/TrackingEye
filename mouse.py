"""macOS cursor control via Quartz event taps.

Everything here goes through CGEventPost, which is what the OS itself uses,
so it works in any app.  It requires the process running this script to have
Accessibility permission (System Settings > Privacy & Security > Accessibility).
"""

import Quartz

_LEFT = Quartz.kCGMouseButtonLeft
_RIGHT = Quartz.kCGMouseButtonRight


def can_post_events():
    """True if this process holds Accessibility permission.

    Without it every cursor event below is silently swallowed by the OS —
    no error, nothing moves — so check before blaming the tracking.
    """
    return bool(Quartz.CGPreflightPostEventAccess())


def request_post_event_access():
    """Ask macOS for Accessibility. Shows the system prompt once per app."""
    return bool(Quartz.CGRequestPostEventAccess())


def screen_size():
    """Main display size in the same coordinate space as the cursor (points)."""
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return bounds.size.width, bounds.size.height


def cursor_position():
    event = Quartz.CGEventCreate(None)
    point = Quartz.CGEventGetLocation(event)
    return point.x, point.y


def _post(event_type, x, y, button=_LEFT, clicks=1):
    event = Quartz.CGEventCreateMouseEvent(None, event_type, (x, y), button)
    if clicks > 1:
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, clicks)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


class Mouse:
    """Stateful cursor driver.  Knows whether a button is currently held."""

    def __init__(self):
        self.width, self.height = screen_size()
        self.left_down = False
        self.right_down = False

    def _clamp(self, x, y):
        return (
            max(0.0, min(self.width - 1.0, x)),
            max(0.0, min(self.height - 1.0, y)),
        )

    def move_by(self, dx, dy):
        """Relative move, trackpad style.  Returns the new position."""
        x, y = cursor_position()
        x, y = self._clamp(x + dx, y + dy)
        if self.left_down:
            _post(Quartz.kCGEventLeftMouseDragged, x, y, _LEFT)
        elif self.right_down:
            _post(Quartz.kCGEventRightMouseDragged, x, y, _RIGHT)
        else:
            _post(Quartz.kCGEventMouseMoved, x, y)
        return x, y

    def left_press(self):
        if self.left_down:
            return
        x, y = cursor_position()
        _post(Quartz.kCGEventLeftMouseDown, x, y, _LEFT)
        self.left_down = True

    def left_release(self):
        if not self.left_down:
            return
        x, y = cursor_position()
        _post(Quartz.kCGEventLeftMouseUp, x, y, _LEFT)
        self.left_down = False

    def left_click(self):
        x, y = cursor_position()
        _post(Quartz.kCGEventLeftMouseDown, x, y, _LEFT)
        _post(Quartz.kCGEventLeftMouseUp, x, y, _LEFT)

    def right_click(self):
        x, y = cursor_position()
        _post(Quartz.kCGEventRightMouseDown, x, y, _RIGHT)
        _post(Quartz.kCGEventRightMouseUp, x, y, _RIGHT)

    def scroll(self, dy_pixels):
        if abs(dy_pixels) < 1:
            return
        event = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitPixel, 1, int(dy_pixels)
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def release_all(self):
        """Panic path: never leave a button stuck down."""
        self.left_release()
        if self.right_down:
            x, y = cursor_position()
            _post(Quartz.kCGEventRightMouseUp, x, y, _RIGHT)
            self.right_down = False
