"""Build TrackingEye.app on the Desktop: draws the icon, writes the bundle.

    .venv/bin/python make_desktop_app.py

Double-clicking the result launches the tracker with no Terminal window. The
bundle is a launcher — the code still lives in this folder, so edits to
config.py take effect on the next launch.
"""

import os
import plistlib
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from tracker import CONNECTIONS

PROJECT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.expanduser("~/Desktop/TrackingEye.app")
SIZE = 1024


def hand_points():
    """The 'pointing' pose, the gesture the app is built around."""
    points = [None] * 21
    points[0] = (0.50, 0.86)
    columns = {"index": 0.44, "middle": 0.52, "ring": 0.60, "pinky": 0.67}
    joints = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
              "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    extended = {"index": True, "middle": False, "ring": False, "pinky": False}
    for name, (mcp, pip, dip, tip) in joints.items():
        x = columns[name]
        points[mcp], points[pip] = (x, 0.60), (x, 0.48)
        points[dip], points[tip] = ((x, 0.36), (x, 0.24)) if extended[name] else \
                                   ((x, 0.44), (x, 0.52))
    points[1], points[2], points[3] = (0.38, 0.76), (0.31, 0.68), (0.27, 0.60)
    points[4] = (0.24, 0.52)
    return points


def draw_icon():
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    # Vertical gradient, dark slate to near-black.
    for y in range(SIZE):
        shade = 46 - int(18 * y / SIZE)
        canvas[y, :] = (shade + 10, shade + 4, shade)

    # Rounded-square mask, macOS style.
    mask = np.zeros((SIZE, SIZE), np.uint8)
    margin, radius = int(SIZE * 0.09), int(SIZE * 0.22)
    cv2.rectangle(mask, (margin + radius, margin), (SIZE - margin - radius, SIZE - margin), 255, -1)
    cv2.rectangle(mask, (margin, margin + radius), (SIZE - margin, SIZE - margin - radius), 255, -1)
    for center in [(margin + radius, margin + radius), (SIZE - margin - radius, margin + radius),
                   (margin + radius, SIZE - margin - radius),
                   (SIZE - margin - radius, SIZE - margin - radius)]:
        cv2.circle(mask, center, radius, 255, -1)

    points = hand_points()
    accent = (80, 220, 255)  # same cyan the app uses for "move"
    for start, end in CONNECTIONS:
        p1 = (int(points[start][0] * SIZE), int(points[start][1] * SIZE))
        p2 = (int(points[end][0] * SIZE), int(points[end][1] * SIZE))
        cv2.line(canvas, p1, p2, accent, 14, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(canvas, (int(x * SIZE), int(y * SIZE)), 22, accent, -1, cv2.LINE_AA)

    # The tracked fingertip, highlighted, with a cursor arrow leaving it.
    tip = (int(points[8][0] * SIZE), int(points[8][1] * SIZE))
    cv2.circle(canvas, tip, 54, (255, 255, 255), 6, cv2.LINE_AA)
    arrow = np.array([[0, 0], [0, 132], [36, 100], [60, 150], [86, 138],
                      [62, 90], [108, 84]], np.int32)
    arrow += np.array([tip[0] + 74, tip[1] - 150], np.int32)
    cv2.fillPoly(canvas, [arrow], (255, 255, 255), cv2.LINE_AA)

    canvas[mask == 0] = 0
    alpha = mask.copy()
    return np.dstack([canvas, alpha])


def build_icns(icon_rgba, destination):
    with tempfile.TemporaryDirectory() as work:
        iconset = os.path.join(work, "AppIcon.iconset")
        os.makedirs(iconset)
        for size in (16, 32, 128, 256, 512):
            for scale, suffix in ((1, ""), (2, "@2x")):
                pixels = size * scale
                resized = cv2.resize(icon_rgba, (pixels, pixels), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(iconset, f"icon_{size}x{size}{suffix}.png"), resized)
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", destination], check=True)


def build_bundle():
    if os.path.exists(APP):
        shutil.rmtree(APP)
    macos = os.path.join(APP, "Contents", "MacOS")
    resources = os.path.join(APP, "Contents", "Resources")
    os.makedirs(macos)
    os.makedirs(resources)

    launcher = os.path.join(macos, "TrackingEye")
    with open(launcher, "w") as handle:
        handle.write("#!/bin/bash\n"
                     f'cd "{PROJECT}" || exit 1\n'
                     'exec ./run.sh\n')
    os.chmod(launcher, 0o755)

    info = {
        "CFBundleName": "TrackingEye",
        "CFBundleDisplayName": "TrackingEye",
        "CFBundleIdentifier": "local.trackingeye",
        "CFBundleExecutable": "TrackingEye",
        "CFBundleIconFile": "AppIcon",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "11.0",
        "NSCameraUsageDescription":
            "TrackingEye watches your hand through the camera to move the cursor.",
        "NSHighResolutionCapable": True,
    }
    with open(os.path.join(APP, "Contents", "Info.plist"), "wb") as handle:
        plistlib.dump(info, handle)

    build_icns(draw_icon(), os.path.join(resources, "AppIcon.icns"))
    subprocess.run(["touch", APP], check=False)
    print(f"built {APP}")


if __name__ == "__main__":
    build_bundle()
