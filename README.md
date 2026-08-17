# TrackingEye

Control the macOS cursor with your hand, through the laptop camera. MediaPipe finds
the hand; the gestures below become real system mouse events, so they work in every
app — not just this one.

macOS only: cursor control goes through Quartz, and camera selection through
AVFoundation.

## Gestures

| Gesture | Action |
| --- | --- |
| Index finger up | Move the cursor, trackpad style |
| Pinch thumb + index | Left click — hold past 0.35s to drag, release to drop |
| Pinch thumb + middle | Right click |
| Index + middle up | Scroll: move your hand up/down |
| Fist or open palm | Clutch — cursor holds still, reposition your hand freely |

Keys in the preview window: **space** arms/disarms cursor control, **c** switches
camera, **q** or **esc** quits.

Movement is relative, like a trackpad: the cursor moves by how much your hand moved,
not by where your hand is. To cross a big screen you sweep, clutch (make a fist),
move back, and sweep again — though with acceleration on, one brisk sweep already
covers the full screen width. Buttons are always released on quit, on disarm, and if
the hand leaves the frame, so nothing stays stuck down.

## Setup

From a fresh clone:

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

Then fetch the hand model (~7.5 MB, not in the repo):

```bash
mkdir -p models && curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Optionally put a launcher on the Desktop:

```bash
.venv/bin/python make_desktop_app.py
```

## Run it

```bash
./run.sh
```

Or double-click **TrackingEye** on the Desktop. That bundle is only a launcher — the
code stays in this folder, so edits to `config.py` apply on its next launch. Because
macOS treats it as its *own* application, it asks for Camera and Accessibility
separately from your terminal.

## Permissions

Both permissions are granted to **whichever app runs the script** (Terminal, iTerm,
VS Code, or the Desktop bundle), never to the script itself. Launch it from a real
app that can show a dialog — from anywhere that can't, macOS denies the camera
outright instead of asking.

1. **Camera** — prompted on first run. The app waits 30s for you to click Allow.
2. **Accessibility** — prompted too, but the grant only takes effect after you
   **restart the app that runs it**. Until then tracking runs and the cursor stays
   put; the preview says so in red rather than leaving you guessing.

If either was ever denied, macOS won't ask again — add the app by hand under System
Settings › Privacy & Security › Camera / Accessibility, then restart it.

## Choosing a camera

Continuity Camera hands your iPhone to anything that asks for "camera 0", and the
phone's viewing angle breaks gesture detection badly: a hand that reads as `move` on
the laptop camera reads as a constant `pinch` from the phone, because the
"fingertip above knuckle" test assumes you're facing the lens. So the app picks the
**built-in camera by device type**, never by index.

```bash
./run.sh --list-cameras     # what's available
./run.sh --pick-camera      # choose interactively at startup
./run.sh --camera 1         # force one
```

Press **c** while running to switch cameras live. To pin a choice permanently, set
`CAMERA_INDEX` in [config.py](config.py).

## Calibration

Hands differ, and the pinch thresholds are the one thing worth measuring rather than
guessing:

```bash
./run.sh --calibrate
```

Hold four poses while a bar fills. It reports how close your thumb actually gets to
each fingertip and recommends `PINCH_CLOSE` / `PINCH_RELEASE`, refusing to recommend
anything when the recording can't support it — a pinch you didn't hold, an
index-pinch step that captured a middle-finger pinch, or a gap too narrow to split
without clicking while you point.

Raw samples are saved to `calibration_samples.json`, so you can re-run the analysis
without waving at the camera again:

```bash
./run.sh --calibrate --reanalyze
```

Measured on this machine: index pinch `0.05`, right-click pinch `0.23`, pointing hand
`0.62` (never closer than `0.53`), fist `0.25`. Two things fall out of those numbers:

- A fist measures *tighter* than a real pinch, so distance alone can't tell them
  apart. A pinch also requires the pinching finger to be extended — otherwise the
  clutch gesture would fire clicks continuously.
- One threshold serves both pinches, so it must clear the looser one. Calibrating
  against the index pinch alone gives `0.24`, which would leave right-click
  (`0.23`) sitting on the line.

## Tuning

Everything adjustable lives in [config.py](config.py); no other file needs editing.

- `SENSITIVITY` — screen pixels per unit of hand movement. Cursor sluggish? Start here.
- `ACCELERATION` — extra gain on fast sweeps, so quick movement crosses the screen
  while slow movement stays precise. `0` gives a uniform, trackpad-flat feel.
- `JITTER_CUTOFF` / `SPEED_COMPENSATION` — the [One Euro filter](filters.py). Lower
  the cutoff for a steadier cursor at rest; raise the compensation if fast movement
  feels laggy. Unlike plain smoothing, these two don't trade against each other.
- `DEADZONE` — ignores hand tremor. Raise it if the cursor creeps while you hold still.
- `PINCH_CLOSE` / `PINCH_RELEASE` — how firm a pinch must be, scaled by hand size so
  it behaves the same near and far from the camera. Set these from calibration.
- `DRAG_HOLD_SECONDS` — a pinch held longer than this is a drag, not a click.
- `START_ARMED` — `False` launches disarmed; take control with space.

Run with `--stats` for a line a second of frame rate, hand-detection rate, and which
gestures fired. It's the fastest way to tell a tracking problem from a tuning one:

```
[stats] 30.0 fps  hand 100%  move:31
[stats] 30.0 fps  hand 100%  move:20 pinch:11
```

## How it works

Each frame: MediaPipe returns 21 hand landmarks → the pose is classified into one of
five gestures → an adaptive filter smooths the anchor point → the movement since the
last frame becomes a relative cursor event.

Three details that aren't obvious:

- **The cursor follows the index knuckle, not the fingertip.** The fingertip shifts
  when you pinch, which would drag the cursor off target at the moment you click.
- **Distances are divided by hand size** (wrist to middle knuckle), so gestures mean
  the same thing whether your hand is near the camera or far from it.
- **Clutching resyncs the filters.** Re-engaging after repositioning would otherwise
  replay the accumulated lag as a cursor fling.

## Layout

| File | |
| --- | --- |
| [main.py](main.py) | Camera loop, cursor state machine, preview overlay |
| [tracker.py](tracker.py) | MediaPipe wrapper and gesture classification |
| [mouse.py](mouse.py) | macOS cursor events and Accessibility checks, via Quartz |
| [filters.py](filters.py) | One Euro filter — steady at rest, responsive when moving |
| [cameras.py](cameras.py) | Device listing and built-in camera detection |
| [calibrate.py](calibrate.py) | Measures your hand, recommends thresholds |
| [config.py](config.py) | All tunable settings |
| [make_desktop_app.py](make_desktop_app.py) | Builds the Desktop launcher and its icon |
| [test_gestures.py](test_gestures.py) | Camera-free tests |

## Tests

```bash
.venv/bin/python test_gestures.py
```

32 checks, no camera needed. Synthetic hand landmarks drive the classifier, the
click/drag/clutch state machine, the pointer filter, and the calibration analysis.
The cases worth knowing about: a fist must not read as a pinch, a held pinch must
become a drag and release cleanly, losing the hand mid-drag must let go of the
button, and re-engaging after a clutch must not fling the cursor.

Measured live on the built-in camera: **30 fps, hand detected in 100% of frames**,
`move` held steadily while pointing with no false pinches, and clicks registering in
bursts of 8–16 frames. The detector costs ~6 ms/frame at 960×540 on an M4, so the
camera's 30 fps is the ceiling, not the tracking.

## Notes

- Pinned to `mediapipe==0.10.35`. Version 1.0.1 crashes on macOS arm64 — its hand
  graph demands a Metal service that isn't available, even with the CPU delegate
  forced. 0.10.35 also dropped `mp.solutions`, hence the Tasks API and the separate
  model file.
- `calibration_samples.json` holds measurements of your hand and is gitignored.
- Right-click and scroll are implemented and unit-tested, but haven't yet been
  confirmed against a live camera the way move, click, and drag have.
