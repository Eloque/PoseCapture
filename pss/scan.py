"""Video -> pose scan: DWPose whole-body on every frame, keeping the 18 OpenPose body points AND the six foot points
(heel, big toe, small toe per foot), which the usual DWPose JSON drops. A side-view walk's ground contact is only
known through heel strike and toe-off with them.

The scan is a dict (also its JSON file format):
  {"video", "crop", "start", "every", "fps", "size": [w, h] (of the scanned frames, after the crop),
   "feet_order": [...six names...], "note",
   "frames": [{"frame": source frame number, "body": [[x, y, score] x 18] (OpenPose order) or None,
               "feet": {name: [x, y, score]} or None}]}
Coordinates are pixels of the (cropped) frame. Per frame the most confident person is kept.
"""
import numpy as np

FEET = ["left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]
SCAN_NOTE = "body = OpenPose-18 order; feet from the same DWPose whole-body run"


def parse_crop(crop):
    """'x,y,w,h' -> (x, y, w, h) ints, '' or '0' -> None"""
    crop = (crop or "").strip()
    if crop in ("", "0", "none", "None"):
        return None
    x, y, w, h = (int(float(v)) for v in crop.split(","))
    return x, y, w, h


def apply_crop(img, crop):
    if crop is None:
        return img
    x, y, w, h = crop
    return img[y:y + h, x:x + w]


def pose_record(info, frame_no):
    """info = Wholebody output (people, 134, 3) for one frame; the neck is inserted at 17, so the six COCO-WholeBody
    foot points (17..22) sit at 18..23"""
    rec = {"frame": frame_no, "body": None, "feet": None}
    if info is not None and len(info):
        p = max(info, key=lambda q: float(q[:18, 2].mean()))   # the most confident person
        rec["body"] = [[round(float(v), 2) for v in p[j]] for j in range(18)]
        rec["feet"] = {name: [round(float(v), 2) for v in p[18 + j]] for j, name in enumerate(FEET)}
    return rec


def scan_frames(frames_bgr, model, frame_numbers, progress=None):
    """frames_bgr: iterable of HxWx3 uint8 BGR images (already cropped); model: a DWPose Wholebody instance"""
    out = []
    for i, (img, n) in enumerate(zip(frames_bgr, frame_numbers)):
        out.append(pose_record(model(np.ascontiguousarray(img)), n))
        if progress:
            progress(i + 1)
    return out


def new_scan(frames, video="", crop=None, start=0, every=1, fps=0.0, size=(0, 0)):
    return {"video": video, "crop": ",".join(str(v) for v in crop) if crop else None, "start": start, "every": every,
            "fps": fps, "size": [int(size[0]), int(size[1])], "feet_order": list(FEET), "note": SCAN_NOTE,
            "frames": frames}


def scan_size(scan):
    """(w, h) of the scanned frames: stored, else from the crop, else the walk scans' 640x720 convention is unknown ->
    the extent of the points"""
    s = scan.get("size")
    if s and s[0] > 0:
        return int(s[0]), int(s[1])
    c = parse_crop(scan.get("crop") or "")
    if c:
        return c[2], c[3]
    pts = np.array([p[:2] for f in scan["frames"] if f["body"] for p in f["body"]], float)
    return int(np.ceil(pts[:, 0].max())) + 1, int(np.ceil(pts[:, 1].max())) + 1


def summary(scan):
    fr = scan["frames"]
    seen = [f for f in fr if f["body"]]
    w, h = scan_size(scan)
    msg = f"{len(fr)} frames scanned ({w}x{h}), a person in {len(seen)}"
    if seen:
        conf = np.array([[p[2] for p in f["body"]] for f in seen])
        msg += f"; mean body score {conf.mean():.2f}, lowest frame {conf.mean(1).min():.2f}"
    return msg
