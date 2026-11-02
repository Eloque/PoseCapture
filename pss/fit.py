"""Fit a skeleton track (OpenPose-18, one person, side view) to a character's proportions.

Logic copied unchanged from the workflow bundle's scripts/fit_track.py; only the file I/O moved out (the node passes
loaded JSON data in and gets the results back as data). The track keeps every bone's DIRECTION per frame but takes the
character's bone LENGTHS (torso, upper arm, forearm, shoulder and hip width, all relative to leg length, measured on
the character's own DWPose keypoints). The whole move is then placed on a canvas with ONE transform (figure height
and ground line of the character picture), and the track is shifted so the hips stay at hip_x (in place), or with
fixed_shift by one shift for the whole clip (walks: the planted feet keep one steady speed).
"""
import numpy as np
from PIL import Image, ImageDraw

NECK, RSH, REL, RWR, LSH, LEL, LWR, RHIP, RKN, RAN, LHIP, LKN, LAN = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
HEAD = [0, 14, 15, 16, 17]
BONES = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0)]
W0, H0 = 768, 1024                         # canvas the character picture lives on


def frames_from_data(frames):
    """frames = loaded DWPose json (list of {"people": [...], "canvas_width", "canvas_height"}), or a POSE_KEYPOINT"""
    out = []
    for f in frames:
        cw, ch = f.get("canvas_width", 1), f.get("canvas_height", 1)
        if not f["people"]:
            out.append(None)
            continue
        r = f["people"][0]["pose_keypoints_2d"]
        k = []
        for i in range(18):
            x, y, c = r[3 * i:3 * i + 3]
            if c <= 0.3:
                k.append(None)
            else:
                k.append((x * cw, y * ch) if x <= 1.0 and y <= 1.0 else (x, y))
        out.append(k)
    return out


def fill(k):
    """side view: a hidden joint sits under its twin"""
    for a, b in ((LSH, RSH), (LEL, REL), (LWR, RWR), (LHIP, RHIP), (LKN, RKN), (LAN, RAN)):
        if k[a] is None: k[a] = k[b]
        if k[b] is None: k[b] = k[a]
    return k


def P(k, i):
    return np.array(k[i], float)


def dist(k, a, b):
    return float(np.linalg.norm(P(k, a) - P(k, b)))


def measures(k):
    leg = (dist(k, RHIP, RKN) + dist(k, RKN, RAN) + dist(k, LHIP, LKN) + dist(k, LKN, LAN)) / 2
    mid = (P(k, RHIP) + P(k, LHIP)) / 2
    eyes = (P(k, 14) + P(k, 15)) / 2
    return {"torso": float(np.linalg.norm(P(k, NECK) - mid)) / leg,
            "uarm": (dist(k, RSH, REL) + dist(k, LSH, LEL)) / 2 / leg,
            "farm": (dist(k, REL, RWR) + dist(k, LEL, LWR)) / 2 / leg,
            "head": float(np.linalg.norm(P(k, NECK) - eyes)) / leg,
            "shw": dist(k, RSH, LSH) / leg, "hipw": dist(k, RHIP, LHIP) / leg}


def retarget(k, s):
    """keep every bone's direction, scale its length by s[bone]; the pelvis is the root"""
    n = [None] * 18
    mid = (P(k, RHIP) + P(k, LHIP)) / 2
    for hip, kn, an in ((RHIP, RKN, RAN), (LHIP, LKN, LAN)):
        n[hip] = mid + (P(k, hip) - mid) * s["hipw"]
        n[kn] = n[hip] + (P(k, kn) - P(k, hip))
        n[an] = n[kn] + (P(k, an) - P(k, kn))
    n[NECK] = mid + (P(k, NECK) - mid) * s["torso"]
    for sh, el, wr in ((RSH, REL, RWR), (LSH, LEL, LWR)):
        n[sh] = n[NECK] + (P(k, sh) - P(k, NECK)) * s["shw"]
        n[el] = n[sh] + (P(k, el) - P(k, sh)) * s["uarm"]
        n[wr] = n[el] + (P(k, wr) - P(k, el)) * s["farm"]
    for h in HEAD:
        n[h] = n[NECK] + (P(k, h) - P(k, NECK)) * s["head"]
    return n


def retarget_track(track_frames, char_frames, scale_frame=0):
    seq = frames_from_data(track_frames)
    prev = None
    for i, k in enumerate(seq):
        k = fill(list(k)) if k else list(prev)
        if prev:
            k = [p if p is not None else q for p, q in zip(k, prev)]
        seq[i] = prev = k
    ck = fill(frames_from_data(char_frames)[0])
    mc = measures(ck)
    # 90th percentile, not mean: side-view bones foreshorten (an arm swinging toward the camera looks short)
    ms = {key: float(np.percentile([measures(k)[key] for k in seq], 90)) for key in ("torso", "uarm", "farm", "head")}
    # the track is a side view: widths are hidden, use a typical human; the character picture is a front view
    ms.update(shw=0.49, hipw=0.33)
    s = {key: mc[key] / ms[key] for key in ms}
    s["head"] = 1.0
    ret = [retarget(k, s) for k in seq]
    ys = [p[1] for p in ck if p is not None]
    ctop, cbot = min(ys), max(ys)
    allp = np.array([p for k in ret for p in k if p is not None])
    f0 = np.array([p for p in ret[min(scale_frame, len(ret) - 1)] if p is not None])
    scale = (cbot - ctop) / (f0[:, 1].max() - f0[:, 1].min())     # figure height from the frame-0 stance
    bot = allp[:, 1].max()
    xs = allp[:, 0]
    cx = (xs.min() + xs.max()) / 2                                # centre the clip's horizontal range on the canvas
    placed = [[(float((p[0] - cx) * scale + W0 / 2), float((p[1] - bot) * scale + cbot)) if p is not None else None
               for p in k] for k in ret]
    return placed, s, mc, ms, ck


def fit_track(track, char_frames, canvas=768, hip_x=384.0, fixed_shift=False, pad_top=0, scale_frame=0):
    """track: dict with "keypoints", "feet", "contact" (+ optional "timing", "cycle"), each the loaded JSON of the
    track folder's file. char_frames: the character's DWPose keypoints (loaded json or POSE_KEYPOINT).
    Returns (fitted track dict with the same keys + "fit", check sheet PIL image, list of warnings, summary text).
    The body is fit_track.py's main() with the json.dump calls replaced by returning the same objects."""
    warnings = []
    placed, s, mc, ms, ck = retarget_track(track["keypoints"], char_frames, scale_frame)
    new = np.array([[p if p is not None else (np.nan, np.nan) for p in k] for k in placed], float)   # (N, 18, 2)
    src_js = track["keypoints"]
    src = np.array([np.array(f["people"][0]["pose_keypoints_2d"], float).reshape(18, 3)[:, :2] for f in src_js])
    feet = track["feet"]["frames"]
    contact = track["contact"]
    N = len(new)
    ANK = {"right": 10, "left": 13}
    KNE = {"right": 9, "left": 12}
    ratio = np.nanmedian([np.linalg.norm(new[t, KNE[sd]] - new[t, ANK[sd]]) / max(np.linalg.norm(src[t, KNE[sd]] - src[t, ANK[sd]]), 1e-6)
                          for t in range(N) for sd in ANK])
    out_feet = []
    for t in range(N):
        fo = {}
        for sd in ANK:
            o = np.array(feet[t][sd]["ankle"])
            fo[sd] = {q: (new[t, ANK[sd]] + (np.array(feet[t][sd][q]) - o) * ratio) for q in ("heel", "toe", "small")}
            fo[sd]["ankle"] = new[t, ANK[sd]].copy()
        out_feet.append(fo)
    hip = (new[:, 8, 0] + new[:, 11, 0]) / 2                      # in place at hip_x
    hip_s = np.convolve(np.pad(hip, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    dx = hip_x - (np.full(N, hip.mean()) if fixed_shift else hip_s)
    new[:, :, 0] += dx[:, None]
    for t in range(N):
        for sd in ANK:
            for q in out_feet[t][sd]:
                out_feet[t][sd][q] = out_feet[t][sd][q] + np.array([dx[t], 0])
    W, H = canvas, H0 + pad_top
    new[:, :, 1] += pad_top
    for t in range(N):
        for sd in ANK:
            for q in out_feet[t][sd]:
                out_feet[t][sd][q] = out_feet[t][sd][q] + np.array([0, pad_top])
    if np.nanmin(new[:, :, 1]) < 40:
        warnings.append(f"WARNING: y min {np.nanmin(new[:, :, 1]):.0f}: the fist leaves the top of the {H}px canvas (pad_top)")
    allx = np.concatenate([new[:, :, 0].ravel()] + [[out_feet[t][sd]["toe"][0] for t in range(N) for sd in ANK]])
    if np.nanmax(allx) > W - 8 or np.nanmin(allx) < 8:
        warnings.append(f"WARNING: x {np.nanmin(allx):.0f}..{np.nanmax(allx):.0f} leaves the {W}px canvas")
    kps = np.concatenate([np.nan_to_num(new), np.where(np.isnan(new[:, :, :1]), 0.0, 1.0)], 2)
    js = [{"people": [{"pose_keypoints_2d": [float(v) for v in k.reshape(-1)]}], "canvas_width": W, "canvas_height": H} for k in kps]
    R75 = 0.75                    # feet.json / contact.json in RENDER px (canvas x 0.75: the renders are 768 px wide)
    near = "right" if "near foot = right" in contact["note"] else "left"
    far = "left" if near == "right" else "right"
    fj, cj = [], []
    for t in range(N):
        fj.append({sd: {q: [round(float(v) * R75, 2) for v in out_feet[t][sd][q]] for q in out_feet[t][sd]} for sd in ANK})
        rec = {"feet": [], "planted": contact["frames"][t]["planted"]}
        for sd in (near, far):
            f = out_feet[t][sd]
            rec["feet"].append([round(float((f["heel"][0] + f["toe"][0]) / 2 * R75), 2), round(float(max(f["heel"][1], f["toe"][1]) * R75), 2)])
        cj.append(rec)
    note = contact["note"].split(";")[0] + "; fitted to the character; render px"
    out = {"keypoints": js, "feet": {"note": note, "frames": fj}, "contact": {"note": note, "frames": cj}}
    for f in ("timing", "cycle"):
        if track.get(f) is not None:
            out[f] = track[f]
    out["fit"] = {"canvas": [W, H], "pad_top": pad_top, "hip_x": hip_x, "scale_frame": scale_frame,
                  "fixed_shift": fixed_shift, "scale_per_bone": s, "feet_ratio": float(ratio)}
    # check sheet: the character's own skeleton, then 6 fitted frames
    picks = np.linspace(0, N - 1, 6).round().astype(int)
    sh = Image.new("RGB", (7 * 256, 342), "white")
    def sk(pts, ox):
        im = Image.new("RGB", (W, H), (20, 20, 20))
        d = ImageDraw.Draw(im)
        for i, j in BONES:
            if pts[i] is not None and pts[j] is not None:
                d.line([tuple(pts[i]), tuple(pts[j])], fill=(255, 200, 80), width=6)
        sh.paste(im.resize((256, 342)), (ox, 0))
    sk(ck, 0)
    for j, i in enumerate(picks):
        sk([tuple(q) for q in new[i]], (j + 1) * 256)
    summary = (f"scale per bone {({k: round(v, 3) for k, v in s.items()})}\n"
               f"{N} frames, canvas {W}x{H}, feet ratio {ratio:.3f}")
    return out, sh, warnings, summary
