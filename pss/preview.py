"""Check pictures for the video nodes: the scan's skeleton over the video, and the cleaned track over the video.

Colours follow the limb-label convention of the workflow: R = near = orange, L = far = blue; head and torso white.
Feet: heel - big toe - small toe triangle; a planted foot gets a green dot at its contact point; the ground line grey.
"""
import numpy as np
from PIL import Image, ImageDraw

NEAR = (255, 140, 0)
FAR = (60, 140, 255)
MID = (235, 235, 235)
PLANT = (40, 220, 90)
BONES_NEAR = [(2, 3), (3, 4), (8, 9), (9, 10)]
BONES_FAR = [(5, 6), (6, 7), (11, 12), (12, 13)]
BONES_MID = [(1, 2), (1, 5), (1, 8), (1, 11), (1, 0), (0, 14), (0, 15), (14, 16), (15, 17)]


def _line_w(h):
    return max(2, round(h / 240))


def draw_pose(img, body, feet_pts=None, planted=None, contact_pts=None, ground=None, alpha_bg=0.55):
    """img: HxWx3 uint8; body: (18, 3) x, y, score; feet_pts: {"right": (heel, toe, small), "left": ...} in px"""
    im = Image.fromarray((img.astype(np.float32) * alpha_bg).astype(np.uint8))
    d = ImageDraw.Draw(im)
    lw = _line_w(img.shape[0])
    if ground is not None:
        d.line([(0, ground), (img.shape[1], ground)], fill=(130, 130, 130), width=max(1, lw // 2))

    def ok(i):
        return body[i][2] > 0.3 and body[i][0] > 1

    for bones, col in ((BONES_MID, MID), (BONES_FAR, FAR), (BONES_NEAR, NEAR)):
        for a, b in bones:
            if ok(a) and ok(b):
                d.line([tuple(body[a][:2]), tuple(body[b][:2])], fill=col, width=lw)
    if feet_pts:
        for side, col, ank in (("left", FAR, 13), ("right", NEAR, 10)):
            if side not in feet_pts:
                continue
            heel, toe, small = feet_pts[side]
            d.polygon([tuple(heel), tuple(toe), tuple(small)], outline=col, width=max(1, lw // 2))
            if ok(ank):
                d.line([tuple(body[ank][:2]), tuple(heel)], fill=col, width=max(1, lw // 2))
    r = lw * 2
    for i in range(18):
        if ok(i):
            x, y = body[i][:2]
            col = NEAR if i in (2, 3, 4, 8, 9, 10) else FAR if i in (5, 6, 7, 11, 12, 13) else MID
            d.ellipse([x - r / 2, y - r / 2, x + r / 2, y + r / 2], fill=col)
    if planted and contact_pts:
        for p, (x, y) in zip(planted, contact_pts):
            if p:
                d.ellipse([x - r, y - r, x + r, y + r], fill=PLANT)
    return np.asarray(im)


def scan_overlay(img, rec, order):
    if rec["body"] is None:
        return (img.astype(np.float32) * 0.55).astype(np.uint8)
    f = rec["feet"]
    fp = {s: (f[f"{s}_heel"][:2], f[f"{s}_big_toe"][:2], f[f"{s}_small_toe"][:2]) for s in ("right", "left")}
    return draw_pose(img, rec["body"], fp)


def track_overlay(img, clean_frame, order, contact_rec, ground):
    """clean_frame: (24, 3) of the cleaned track (body + feet, canvas px); contact_rec: the frame's contact.json record
    (mannequin-frame px, converted back)"""
    idx = {n: 18 + i for i, n in enumerate(order)}
    fp = {s: (clean_frame[idx[f"{s}_heel"], :2], clean_frame[idx[f"{s}_big_toe"], :2],
              clean_frame[idx[f"{s}_small_toe"], :2]) for s in ("right", "left")}
    pts = [(x / 0.9, (y + 4) / 0.9) for x, y in contact_rec["feet"]]
    body = [[float(v) for v in p] for p in clean_frame[:18]]
    return draw_pose(img, body, fp, contact_rec["planted"], pts, ground)


def fit_height(frames, max_h):
    """downscale a list of same-size frames to max_h (0 = keep)"""
    if not frames or max_h <= 0 or frames[0].shape[0] <= max_h:
        return frames
    h, w = frames[0].shape[:2]
    nw = max(2, round(w * max_h / h))
    return [np.asarray(Image.fromarray(f).resize((nw, max_h), Image.LANCZOS)) for f in frames]
