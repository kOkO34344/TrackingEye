"""Measure your hand and recommend pinch thresholds for config.py.

    ./run.sh --calibrate          (or: .venv/bin/python calibrate.py)

Hold each pose while the bar fills. It records how close your thumb actually
gets to each fingertip, then picks thresholds that separate your pinch from
your resting hand — the defaults are guesses, and a thumb that rests near the
index finger will otherwise read as a permanent click.
"""

import json
import os
import sys
import time

import cv2
import numpy as np

import cameras
import config
from tracker import HandTracker

READY_SECONDS = 3.0
SAMPLE_SECONDS = 5.0
SETTLE = 0.25  # ignore the first quarter of each recording, while the pose settles

POSES = [
    ("point", "POINT: index up, thumb OUT to the side",
     "hold it still — thumb clear of the index finger"),
    ("pinch", "PINCH: thumb hard against INDEX fingertip",
     "hold it shut the whole time, don't release"),
    ("middle", "RIGHT PINCH: thumb against MIDDLE fingertip",
     "hold it shut; keep the index finger out of the way"),
    ("fist", "FIST: hand closed",
     "the clutch pose"),
]


def banner(frame, title, subtitle, progress, sampling):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 96), (24, 24, 24), -1)
    color = (90, 255, 120) if sampling else (80, 200, 255)
    cv2.putText(frame, title, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    label = "recording..." if sampling else "get ready"
    cv2.putText(frame, f"{label}   {subtitle}", (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (16, 80), (16 + int((width - 32) * progress), 88), color, -1)
    return frame


def collect(camera, tracker, start_time):
    """Run the pose sequence, returning {pose: {"index": [...], "middle": [...]}}."""
    samples = {name: {"index": [], "middle": []} for name, _t, _s in POSES}

    for name, title, subtitle in POSES:
        phase_start = time.monotonic()
        while True:
            elapsed = time.monotonic() - phase_start
            if elapsed >= READY_SECONDS + SAMPLE_SECONDS:
                break
            ok, frame = camera.read()
            if not ok:
                continue

            hand = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   int((time.monotonic() - start_time) * 1000))
            sampling = elapsed >= READY_SECONDS
            settled = elapsed >= READY_SECONDS + SAMPLE_SECONDS * SETTLE
            if settled and hand.found:
                m = HandTracker.measure(hand.points)
                samples[name]["index"].append(m["index_ratio"])
                samples[name]["middle"].append(m["middle_ratio"])

            progress = ((elapsed - READY_SECONDS) / SAMPLE_SECONDS if sampling
                        else elapsed / READY_SECONDS)
            view = banner(cv2.flip(frame, 1), title, subtitle, min(progress, 1.0), sampling)
            if not hand.found:
                cv2.putText(view, "no hand detected", (16, view.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 255), 2, cv2.LINE_AA)
            cv2.imshow("TrackingEye — calibration", view)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                return None
    return samples


def summarize(samples):
    print("\nmeasured thumb-to-fingertip distance (relative to hand size)")
    print(f"  {'pose':8s} {'n':>4s}  {'index: p10':>11s} {'median':>8s} {'p90':>7s}"
          f"   {'middle: p10':>12s} {'median':>8s}")
    for name, _title, _sub in POSES:
        index = np.array(samples[name]["index"])
        middle = np.array(samples[name]["middle"])
        if len(index) == 0:
            print(f"  {name:8s}    0   (no hand detected)")
            continue
        print(f"  {name:8s} {len(index):4d}  {np.percentile(index, 10):11.2f} "
              f"{np.median(index):8.2f} {np.percentile(index, 90):7.2f}   "
              f"{np.percentile(middle, 10):12.2f} {np.median(middle):8.2f}")
        # ~30 fps over the part of the window that is actually recorded.
        expected = SAMPLE_SECONDS * (1 - SETTLE) * 30
        if len(index) < 0.5 * expected:
            print(f"           ^ hand found in only ~{len(index) / expected:.0%} of "
                  "frames — keep your whole hand in view, palm toward the camera")


def split_clusters(values):
    """Split samples into low and high groups at the most natural boundary.

    People pinch and release during the recording rather than holding one
    distance, so the samples are two clusters. Take the split that minimizes
    variance within each side (1-D Otsu), and keep the closed one.
    """
    ordered = np.sort(values)
    best_at, best_score = len(ordered), None
    for cut in range(1, len(ordered)):
        low, high = ordered[:cut], ordered[cut:]
        score = low.var() * len(low) + high.var() * len(high)
        if best_score is None or score < best_score:
            best_at, best_score = cut, score
    return ordered[:best_at], ordered[best_at:]


MIN_MARGIN = 0.12  # smallest usable gap between a closed pinch and a pointing hand


def closed_samples(values):
    """The frames where the pinch was shut, plus what fraction of the take that was.

    A steadily-held pinch is a single cluster, and splitting it would report a
    bogus 50% "release" — so only treat the split as real when the two halves
    are genuinely far apart relative to their own spread.
    """
    low, high = split_clusters(values)
    if len(low) == 0 or len(high) == 0:
        return values, 1.0
    separation = float(np.median(high) - np.median(low))
    spread = max(float(low.std()), float(high.std()), 1e-6)
    if separation < 0.15 or separation < 2.5 * spread:
        return values, 1.0
    return low, len(low) / len(values)


def evaluate(samples):
    """Analyse the recording. Pure: returns a verdict, prints nothing.

    status is one of: insufficient, wrong-finger, not-held, too-close, ok.
    """
    pinch = np.asarray(samples["pinch"]["index"])
    pinch_middle = np.asarray(samples["pinch"]["middle"])
    point = np.asarray(samples["point"]["index"])
    fist = np.asarray(samples["fist"]["index"])
    right = np.asarray(samples["middle"]["middle"])

    if len(pinch) < 10 or len(point) < 10:
        return {"status": "insufficient", "pinch_n": len(pinch), "point_n": len(point)}

    # Did that step capture a pinch of the *index* finger, or another one?
    if len(pinch_middle) and np.median(pinch_middle) < np.median(pinch):
        return {"status": "wrong-finger", "index": float(np.median(pinch)),
                "middle": float(np.median(pinch_middle))}

    pinched, held = closed_samples(pinch)
    if held < 0.5:
        released = pinch[pinch > np.max(pinched)]
        return {"status": "not-held", "held": held, "frames": len(pinched),
                "total": len(pinch), "released_at": float(np.median(released))}

    # One threshold serves both pinches, so it has to clear the looser of the two —
    # a right-click pinch never closes as tightly as an index pinch.
    closed = float(np.percentile(pinched, 90))
    right_closed = None
    if len(right) >= 10:
        right_pinched, right_held = closed_samples(right)
        if right_held >= 0.5:
            right_closed = float(np.percentile(right_pinched, 90))
            closed = max(closed, right_closed)

    open_ = float(np.percentile(point, 10))      # closest a *pointing* thumb ever got
    verdict = {
        "closed": closed,
        "closed_index": float(np.percentile(pinched, 90)),
        "closed_right": right_closed,
        "open": open_,
        "typical_open": float(np.median(point)),
        "held": held,
        "margin": open_ - closed,
        "fist": float(np.median(fist)) if len(fist) else None,
        "right_pinch": float(np.median(right)) if len(right) else None,
    }
    if verdict["margin"] < MIN_MARGIN:
        verdict["status"] = "too-close"
        return verdict

    verdict["status"] = "ok"
    verdict["pinch_close"] = closed + verdict["margin"] * 0.35
    verdict["pinch_release"] = closed + verdict["margin"] * 0.70
    return verdict


def report(v):
    status = v["status"]
    if status == "insufficient":
        print("\nNot enough samples to recommend thresholds — rerun and keep your "
              "hand in frame for the whole recording.")
        return
    if status == "wrong-finger":
        print(f"\n!! During the index-pinch step your thumb sat closer to the MIDDLE "
              f"fingertip ({v['middle']:.2f})\n   than the index one ({v['index']:.2f}). "
              "That step recorded the wrong gesture — rerun and\n   touch thumb to "
              "*index* fingertip, keeping the middle finger clear.")
        return
    if status == "not-held":
        print(f"\n!! The pinch was only closed for {v['held']:.0%} of the recording "
              f"({v['frames']} of {v['total']} frames);\n   the rest sat at "
              f"{v['released_at']:.2f}. Hold the pinch shut for the whole green bar and "
              "rerun —\n   a partly-held pinch makes the threshold look tighter than "
              "it is.")
        return

    print(f"\nindex pinch closes to {v['closed_index']:.2f} (held {v['held']:.0%} of "
          "the recording)")
    if v.get("closed_right") is not None:
        print(f"right-click pinch closes to {v['closed_right']:.2f} — the threshold has "
              "to clear this too")
    print(f"pointing hand sits at {v['typical_open']:.2f}, but came as close as "
          f"{v['open']:.2f}")

    if status == "too-close":
        print(f"\n!! Only {v['margin']:.2f} separates your pinch from your pointing hand "
              "— too little to\n   split reliably; a threshold in there would click while "
              "you point. Rerun, holding\n   the pinch fully shut, and point with the "
              "thumb held clear of the index finger\n   (out to the side, not tucked "
              "under it).")
        return

    print("\nrecommended in config.py:")
    print(f"  PINCH_CLOSE = {v['pinch_close']:.2f}      # was {config.PINCH_CLOSE}")
    print(f"  PINCH_RELEASE = {v['pinch_release']:.2f}    # was {config.PINCH_RELEASE}")
    print(f"\n  (margin: {v['margin']:.2f} between a closed pinch and your closest "
          "pointing frame)")

    if v["fist"] is not None and v["fist"] < v["pinch_close"]:
        print(f"\n  note: your fist measures {v['fist']:.2f}, inside the pinch range. "
              "The classifier\n  ignores it because a fist curls the index finger, so "
              "this is fine.")
    if v["right_pinch"] is not None and v["right_pinch"] > v["pinch_close"]:
        print(f"\n  note: your right-click pinch only reaches {v['right_pinch']:.2f}, "
              "above the threshold.\n  Touch thumb and middle fingertip more firmly, or "
              "raise PINCH_CLOSE.")


def recommend(samples):
    report(evaluate(samples))


SAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration_samples.json")


def main():
    # Re-run the analysis on the last recording, without waving at the camera again.
    if "--reanalyze" in sys.argv:
        with open(SAMPLES_PATH) as handle:
            saved = json.load(handle)
        print(f"re-analyzing {SAMPLES_PATH}")
        summarize(saved)
        recommend(saved)
        return

    index = int(sys.argv[sys.argv.index("--camera") + 1]) if "--camera" in sys.argv else (
        config.CAMERA_INDEX if config.CAMERA_INDEX is not None else cameras.builtin_index())
    print(f"camera {index}: {cameras.camera_name(index)}")

    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not camera.isOpened():
        sys.exit(f"Could not open camera {index}. Available:\n{cameras.describe()}")

    tracker = HandTracker()
    print(__doc__)
    try:
        samples = collect(camera, tracker, time.monotonic())
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()

    if samples is None:
        print("cancelled")
        return

    with open(SAMPLES_PATH, "w") as handle:
        json.dump(samples, handle)
    print(f"\nraw samples saved to {os.path.basename(SAMPLES_PATH)} "
          "(re-analyze with --reanalyze)")

    summarize(samples)
    recommend(samples)


if __name__ == "__main__":
    main()
