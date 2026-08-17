"""Camera selection.

macOS Continuity Camera inserts a nearby iPhone into the video device list, and
it can land at index 0 — so a hardcoded index will silently grab your phone
instead of the laptop. Pick the built-in camera by device type instead.
"""

BUILTIN = "AVCaptureDeviceTypeBuiltInWideAngleCamera"
DESK_VIEW = "AVCaptureDeviceTypeDeskViewCamera"


def list_cameras():
    """[(index, name, type)] in the order OpenCV's AVFoundation backend indexes."""
    try:
        import AVFoundation as AV
    except ImportError:
        return []
    devices = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo) or []
    return [(i, d.localizedName(), d.deviceType()) for i, d in enumerate(devices)]


def builtin_index(default=0):
    """Index of the laptop's own camera, or `default` if it can't be identified."""
    for index, _name, kind in list_cameras():
        if kind == BUILTIN:
            return index
    return default


def camera_name(index):
    for i, name, _kind in list_cameras():
        if i == index:
            return name
    return f"camera {index}"


def describe():
    cameras = list_cameras()
    if not cameras:
        return "  (could not enumerate cameras; falling back to index 0)"
    lines = []
    for index, name, kind in cameras:
        tag = ""
        if kind == BUILTIN:
            tag = "  <- built-in, used by default"
        elif kind == DESK_VIEW:
            tag = "  (Desk View)"
        elif "Continuity" in kind or "External" in kind:
            tag = "  (Continuity/external - e.g. your iPhone)"
        lines.append(f"  {index}: {name}{tag}")
    return "\n".join(lines)


def next_index(current):
    """The next camera in the list, wrapping around. Used by the `c` hotkey."""
    indexes = [i for i, _name, _kind in list_cameras()]
    if not indexes:
        return current
    if current not in indexes:
        return indexes[0]
    return indexes[(indexes.index(current) + 1) % len(indexes)]


def prompt_for_camera(default):
    """Ask in the terminal which camera to use. Enter alone keeps the default."""
    found = list_cameras()
    if not found:
        print("Could not enumerate cameras; using index 0.")
        return default
    print("\nAvailable cameras:")
    print(describe())
    while True:
        answer = input(f"Camera index [{default}]: ").strip()
        if not answer:
            return default
        if answer.isdigit() and any(i == int(answer) for i, _n, _k in found):
            return int(answer)
        print(f"Pick one of: {', '.join(str(i) for i, _n, _k in found)}")


if __name__ == "__main__":
    print(describe())
