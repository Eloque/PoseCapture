"""Pose scan -> skeleton track (the format Load Skeleton Track reads and Fit Track to Character takes).

Conventions of the track: OpenPose-18 body points per frame, the figure faces RIGHT, and the R side (2, 3, 4, 8, 9,
10) is the NEAR side (toward the camera). So a scan is first turned into that: mirrored if the person faces left, and
its limb labels swapped if DWPose called the near side "left".

Two cleanups, both "foot locking" (one ground line for both feet, planted points that do not slide or jitter):
- walk cycle: one cycle start..start+period-1, closed into a seamless loop; planted points move back at exactly one
  treadmill speed. This is the workflow's clean_walk.py, logic unchanged. A walk across the frame is first put on a
  treadmill (the body's travel subtracted).
- one-shot (in place): any other move (punch, kick, step-in attack); planted points stay exactly still.

Contact per heel/toe point: it is "down" while near its side's ground (within ground_tol px) and not swinging
forward (forward speed <= swing_tol px/frame); a foot is planted while its heel or toe is down. The corrections fade
in and out over two frames either side of a stance, so the swing keeps the video's motion. The far foot plants
higher on screen (perspective); it is moved down onto the near foot's ground line (the knee half as far).
"""
import numpy as np

# ---------------------------------------------------------------- layout
SWAP_BODY = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)]
SWAP_FEET = [("right_big_toe", "left_big_toe"), ("right_small_toe", "left_small_toe"), ("right_heel", "left_heel")]
# the walk scan the tolerances were tuned on: figure 525.8 px high, ground_tol 12 px, swing_tol 4 px
REF_FIGURE_H = 525.83
REF_GROUND_TOL = 12.0
REF_SWING_TOL = 4.0
TO_MQ = lambda x, y: (x * 0.9, y * 0.9 - 4)   # canvas -> the 576x640 mannequin frame contact.json has always used


def sides(order):
    idx = {n: 18 + i for i, n in enumerate(order)}
    return {"right": {"heel": idx["right_heel"], "toe": idx["right_big_toe"], "small": idx["right_small_toe"],
                      "ankle": 10, "knee": 9},
            "left": {"heel": idx["left_heel"], "toe": idx["left_big_toe"], "small": idx["left_small_toe"],
                     "ankle": 13, "knee": 12}}


def scan_array(scan):
    """(T, 24, 3) float: body 0..17, feet 18..23 in feet_order; frames without a person -> NaN (repair the scan first)"""
    order = scan["feet_order"]
    rows = []
    for f in scan["frames"]:
        if f["body"] is None:
            rows.append(np.full((24, 3), np.nan))
        else:
            rows.append(np.array(f["body"] + [f["feet"][n] for n in order], float))
    return np.array(rows)


def figure_height(K):
    top = np.nanmin(K[:, [0, 14, 15, 16, 17], 1], axis=1)
    bot = np.nanmax(K[:, 18:, 1], axis=1)
    return float(np.nanmedian(bot - top))


def facing(K, order):
    """+1 = faces right, -1 = faces left: toes ahead of the heels (the nose ahead of the ears as a fallback)"""
    S = sides(order)
    d = []
    for s in S.values():
        ok = (K[:, s["toe"], 2] > 0.3) & (K[:, s["heel"], 2] > 0.3)
        d.extend(K[ok, s["toe"], 0] - K[ok, s["heel"], 0])
    if len(d) >= 3 and abs(np.median(d)) > 1:
        return 1 if np.median(d) > 0 else -1
    ears = K[:, 16:18, 0].mean(1)
    return 1 if np.nanmedian(K[:, 0, 0] - ears) >= 0 else -1


def mirror(K, width):
    M = K.copy()
    M[:, :, 0] = (width - 1) - M[:, :, 0]
    return M


def ground_low(K, order):
    S = sides(order)
    return {s: float(np.nanpercentile(np.maximum(K[:, d["heel"], 1], K[:, d["toe"], 1]), 90)) for s, d in S.items()}


def swap_labels(K, order):
    M = K.copy()
    idx = {n: 18 + i for i, n in enumerate(order)}
    for a, b in SWAP_BODY + [(idx[r], idx[l]) for r, l in SWAP_FEET]:
        M[:, [a, b]] = M[:, [b, a]]
    return M


def travel_speed(K):
    """the body's travel in px/frame: least-squares slope of the hip centre's x"""
    hip = (K[:, 8, 0] + K[:, 11, 0]) / 2
    t = np.arange(len(hip))
    ok = ~np.isnan(hip)
    return float(np.polyfit(t[ok], hip[ok], 1)[0]) if ok.sum() > 2 else 0.0


def smooth(K, sigma):
    """gaussian smoothing in time of the x, y of every point (edges reflected); sigma in frames, 0 = off"""
    if sigma <= 0:
        return K
    r = int(np.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    out = K.copy()
    T = len(K)
    idx = np.arange(T)
    acc = np.zeros_like(K[:, :, :2])
    for o, wt in zip(range(-r, r + 1), w):
        j = np.abs(idx + o)
        j = np.where(j > T - 1, 2 * (T - 1) - j, j).clip(0, T - 1)
        acc += wt * K[j, :, :2]
    out[:, :, :2] = acc
    return out


# ---------------------------------------------------------------- walk cycle (clean_walk.py, logic unchanged)
def clean_walk_cycle(K, order, S, P, frames=57, far_drop=None, ground_tol=REF_GROUND_TOL, swing_tol=REF_SWING_TOL):
    if (frames - 1) % 4 or frames < 2 * P + 3:
        raise ValueError(f"walk cycle: frames ({frames}) must be 4n+1 and > 2 cycles + 2 (period {P})")
    if S + P >= len(K):
        raise ValueError(f"walk cycle: start {S} + period {P} needs frame {S + P}, the scan has {len(K)} frames")
    SIDE = sides(order)
    cyc = K[S:S + P].copy()
    err = K[S + P, :, :2] - K[S, :, :2]
    for t in range(P):
        cyc[t, :, :2] -= err * t / P

    low = {s: np.percentile(np.maximum(cyc[:, d["heel"], 1], cyc[:, d["toe"], 1]), 90) for s, d in SIDE.items()}
    near = max(low, key=low.get)
    far = min(low, key=low.get)
    GROUND = low[near]
    persp = low[near] - low[far]
    far_extra = 0.0
    if far_drop is not None:
        far_extra = far_drop - persp
        persp = far_drop

    dx = {}
    down = {}
    for s, d in SIDE.items():
        for p in ("heel", "toe"):
            x = cyc[:, d[p], 0]
            y = cyc[:, d[p], 1]
            v = (np.roll(x, -1) - np.roll(x, 1)) / 2
            dx[(s, p)] = v
            down[(s, p)] = (y >= low[s] - ground_tol) & (v <= swing_tol)
        for p in ("heel", "toe"):
            m = down[(s, p)]
            down[(s, p)] = m | (np.roll(m, 1) & np.roll(m, -1))
    downs = [dx[k][down[k]] for k in down]
    if not sum(len(v) for v in downs):
        raise ValueError("walk cycle: no planted foot found (check ground_tol / swing_tol, or the move is not a walk)")
    speed = float(np.median(-np.concatenate(downs)))

    def runs(mask):
        if mask.all():
            return [list(range(P))]
        start = int(np.argmin(mask))
        out, cur = [], []
        for k in range(P):
            t = (start + k) % P
            if mask[t]:
                cur.append(t)
            elif cur:
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return out

    clean = cyc.copy()
    corr = {s: np.zeros((P, 2)) for s in SIDE}
    weight = {s: np.zeros(P) for s in SIDE}
    point_target = {}
    for s, d in SIDE.items():
        dy_side = persp if s == far else 0.0
        for p in ("heel", "toe"):
            for run in runs(down[(s, p)]):
                n = np.arange(len(run))
                xs = cyc[run, d[p], 0]
                c = float(np.mean(xs + speed * n))
                for i, t in enumerate(run):
                    point_target[(s, p, t)] = (c - speed * i, GROUND + (far_extra if s == far else 0.0))
        for t in range(P):
            tg = [(p, point_target[(s, p, t)]) for p in ("heel", "toe") if (s, p, t) in point_target]
            if tg:
                corr[s][t] = np.mean([[tx - cyc[t, d[p], 0], ty - cyc[t, d[p], 1]] for p, (tx, ty) in tg], 0)
                weight[s][t] = 1.0
        w0, c0 = weight[s].copy(), corr[s].copy()
        for t in range(P):
            if w0[t] == 0:
                for off, wt in ((1, 0.66), (2, 0.33)):
                    for nb in ((t - off) % P, (t + off) % P):
                        if w0[nb] == 1 and wt > weight[s][t]:
                            weight[s][t] = wt
                            corr[s][t] = c0[nb]
        for t in range(P):
            shift = corr[s][t] * weight[s][t] + np.array([0.0, dy_side * (1 - weight[s][t])])
            for q in ("heel", "toe", "small", "ankle"):
                clean[t, d[q], :2] += shift
            clean[t, d["knee"], :2] += shift * 0.5
            for p in ("heel", "toe"):
                if (s, p, t) in point_target:
                    clean[t, d[p], :2] = point_target[(s, p, t)]

    src = [t % P for t in range(frames)]
    seq = np.array([clean[c, :18] for c in src])
    contact, feet_out = _contact_out(clean, src, SIDE, near, far, point_target)
    note = (f"walk cycle frames {S}..{S + P - 1}: near foot = {near}, ground line y {GROUND:.1f} (canvas) "
            f"= {TO_MQ(0, GROUND)[1]:.1f} (mannequin frame); far foot moved down {persp:.1f} px; treadmill speed "
            f"{speed:.2f} px/frame (canvas) = {speed * 0.9:.2f} (mannequin frame)")
    cycle = {"start": S, "source_period": P, "period": P, "frames": frames, "far_drop": float(persp),
             "planted_near": "".join("#" if contact[t]["planted"][0] else "." for t in range(P)),
             "planted_far": "".join("#" if contact[t]["planted"][1] else "." for t in range(P))}
    return {"seq": seq, "clean": clean, "src": src, "contact": contact, "feet": feet_out, "note": note,
            "cycle": cycle, "near": near, "ground": float(GROUND), "persp": float(persp), "speed": speed}


def _contact_out(clean, src, SIDE, near, far, point_target):
    contact, feet_out = [], []
    for c in src:
        rec, fo = {"feet": [], "planted": []}, {}
        for s in (near, far):     # foot 0 = near, 1 = far
            d = SIDE[s]
            pts = [point_target[(s, p, c)] for p in ("heel", "toe") if (s, p, c) in point_target]
            if pts:
                x, y = np.mean(pts, 0)
            else:
                x, y = (clean[c, d["heel"], 0] + clean[c, d["toe"], 0]) / 2, max(clean[c, d["heel"], 1], clean[c, d["toe"], 1])
            mx, my = TO_MQ(x, y)
            rec["feet"].append([round(float(mx), 2), round(float(my), 2)])
            rec["planted"].append(bool(pts))
            fo[s] = {q: [round(float(v), 2) for v in clean[c, d[q], :2]] for q in ("heel", "toe", "small", "ankle")}
        contact.append(rec)
        feet_out.append(fo)
    return contact, feet_out


# ---------------------------------------------------------------- one-shot, in place
def clean_one_shot(K, order, far_drop=None, ground_tol=REF_GROUND_TOL, swing_tol=REF_SWING_TOL):
    """K = the window of the move (T, 24, 3). Planted points are pinned to their run's mean x on the ground line."""
    SIDE = sides(order)
    T = len(K)
    win = K.copy()
    low = {s: np.percentile(np.maximum(win[:, d["heel"], 1], win[:, d["toe"], 1]), 90) for s, d in SIDE.items()}
    near = max(low, key=low.get)
    far = min(low, key=low.get)
    GROUND = low[near]
    persp = low[near] - low[far]
    far_extra = 0.0
    if far_drop is not None:
        far_extra = far_drop - persp
        persp = far_drop

    down = {}
    for s, d in SIDE.items():
        for p in ("heel", "toe"):
            x = win[:, d[p], 0]
            y = win[:, d[p], 1]
            v = np.gradient(x) if T > 1 else np.zeros(T)
            down[(s, p)] = (y >= low[s] - ground_tol) & (np.abs(v) <= swing_tol)
        for p in ("heel", "toe"):
            m = down[(s, p)]
            hole = np.zeros(T, bool)
            hole[1:-1] = m[:-2] & m[2:]
            down[(s, p)] = m | hole

    def runs(mask):
        out, cur = [], []
        for t in range(T):
            if mask[t]:
                cur.append(t)
            elif cur:
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return out

    clean = win.copy()
    point_target = {}
    for s, d in SIDE.items():
        dy_side = persp if s == far else 0.0
        corr = np.zeros((T, 2))
        weight = np.zeros(T)
        for p in ("heel", "toe"):
            for run in runs(down[(s, p)]):
                c = float(np.mean(win[run, d[p], 0]))
                for t in run:
                    point_target[(s, p, t)] = (c, GROUND + (far_extra if s == far else 0.0))
        for t in range(T):
            tg = [(p, point_target[(s, p, t)]) for p in ("heel", "toe") if (s, p, t) in point_target]
            if tg:
                corr[t] = np.mean([[tx - win[t, d[p], 0], ty - win[t, d[p], 1]] for p, (tx, ty) in tg], 0)
                weight[t] = 1.0
        w0, c0 = weight.copy(), corr.copy()
        for t in range(T):
            if w0[t] == 0:
                for off, wt in ((1, 0.66), (2, 0.33)):
                    for nb in (t - off, t + off):
                        if 0 <= nb < T and w0[nb] == 1 and wt > weight[t]:
                            weight[t] = wt
                            corr[t] = c0[nb]
        for t in range(T):
            shift = corr[t] * weight[t] + np.array([0.0, dy_side * (1 - weight[t])])
            for q in ("heel", "toe", "small", "ankle"):
                clean[t, d[q], :2] += shift
            clean[t, d["knee"], :2] += shift * 0.5
            for p in ("heel", "toe"):
                if (s, p, t) in point_target:
                    clean[t, d[p], :2] = point_target[(s, p, t)]

    src = list(range(T))
    contact, feet_out = _contact_out(clean, src, SIDE, near, far, point_target)
    note = (f"one-shot in place: near foot = {near}, ground line y {GROUND:.1f} (canvas); far foot moved down "
            f"{persp:.1f} px; planted points held still")
    return {"seq": clean[:, :18].copy(), "clean": clean, "src": src, "contact": contact, "feet": feet_out,
            "note": note, "cycle": None, "near": near, "ground": float(GROUND), "persp": float(persp), "speed": 0.0}


# ---------------------------------------------------------------- loop finder (find_loop.py)
def _norm(k):
    neck = k[1, :2]
    hips = [k[i, :2] for i in (8, 11) if k[i, 2] > 0.3]
    s = np.linalg.norm(np.mean(hips, 0) - neck) if hips else 1.0
    return (k[:, :2] - neck) / max(s, 1e-3), k[:, 2] > 0.3


def pose_dist(a, b):
    (pa, va), (pb, vb) = _norm(a), _norm(b)
    v = va & vb
    v[14:] = False
    return float(np.mean(np.linalg.norm(pa[v] - pb[v], axis=1))) if v.any() else 9.9


def find_period(K, start, min_len=12, max_len=60, top=5):
    """best b - a for a = start: the frame whose body pose matches the start frame best"""
    res = []
    for j in range(start + min_len, min(start + max_len, len(K) - 1) + 1):
        if not np.isnan(K[start, :18, :2]).any() and not np.isnan(K[j, :18, :2]).any():
            res.append((pose_dist(K[start, :18], K[j, :18]), j - start))
    res.sort()
    return res[:top]


# ---------------------------------------------------------------- the whole step
def make_track(scan, mode="walk cycle", start=0, frames=0, period=0, face="auto", near_side="auto",
               remove_travel="auto", far_drop=-1.0, ground_tol=0.0, swing_tol=0.0, smooth_sigma=0.0):
    """returns (track dict for the Fit node, info text, preview data)"""
    order = scan["feet_order"]
    from .scan import scan_size
    W, H = scan_size(scan)
    K = scan_array(scan)
    if np.isnan(K).any():
        raise ValueError("the scan has frames without a person: run Repair Pose Scan first")
    info = []
    fh = figure_height(K)
    gtol = ground_tol if ground_tol > 0 else REF_GROUND_TOL * fh / REF_FIGURE_H
    stol = swing_tol if swing_tol > 0 else REF_SWING_TOL * fh / REF_FIGURE_H
    info.append(f"scan {len(K)} frames {W}x{H}; figure about {fh:.0f} px high; ground_tol {gtol:.2f} px, "
                f"swing_tol {stol:.2f} px/frame")

    f = facing(K, order) if face == "auto" else (1 if face == "right" else -1)
    mirrored = f < 0
    if mirrored:
        K = mirror(K, W)
    info.append(f"faces {'left: mirrored to face right' if mirrored else 'right'}")

    low = ground_low(K, order)
    near = max(low, key=low.get) if near_side == "auto" else near_side
    swapped = near == "left"
    if swapped:
        K = swap_labels(K, order)
    info.append(f"near side = DWPose '{near}' (feet plant {abs(low['right'] - low['left']):.1f} px apart in depth)"
                + ("; labels swapped so the near side is R" if swapped else ""))

    if smooth_sigma > 0:
        K = smooth(K, smooth_sigma)
        info.append(f"smoothed in time, sigma {smooth_sigma} frames")
    fd = None if far_drop < 0 else far_drop

    if mode == "walk cycle":
        if period <= 0:
            cands = find_period(K, start)
            if not cands:
                raise ValueError("walk cycle: no loop period found after the start frame")
            period = cands[0][1]
            info.append("period auto: " + ", ".join(f"{p} (dist {d:.3f})" for d, p in cands))
        if frames <= 0:
            frames = 4 * ((2 * period + 3 + 3) // 4) + 1
        v = travel_speed(K[start:start + period + 1])
        travel = remove_travel == "on" or (remove_travel == "auto" and abs(v) > 0.01 * fh)
        if travel:
            t = np.arange(len(K)) - start
            K = K.copy()
            K[:, :, 0] -= v * t[:, None]
            info.append(f"travel {v:.2f} px/frame removed (put on a treadmill)")
        else:
            info.append(f"travel {v:.2f} px/frame kept (walk in place)")
        res = clean_walk_cycle(K, order, start, period, frames, fd, gtol, stol)
        src_frames = [start + c for c in res["src"]]
    else:
        end = len(K) if frames <= 0 else start + frames
        if end > len(K):
            raise ValueError(f"one-shot: frames {start}..{end - 1} asked, the scan has {len(K)}")
        res = clean_one_shot(K[start:end], order, fd, gtol, stol)
        src_frames = [start + c for c in res["src"]]
        if (len(src_frames) - 1) % 4:
            info.append(f"note: {len(src_frames)} frames is not 4n+1 (SCAIL/Wan clips want 17, 21, ... 65, 81)")

    seq = res["seq"]
    kp = [{"people": [{"pose_keypoints_2d": [float(v) for v in k.reshape(-1)]}], "canvas_width": W, "canvas_height": H}
          for k in seq]
    note = res["note"]
    track = {"name": "video_track", "keypoints": kp, "feet": {"note": note, "frames": res["feet"]},
             "contact": {"note": note, "frames": res["contact"]}}
    if res["cycle"] is not None:
        track["cycle"] = dict(res["cycle"], fps=scan.get("fps") or 25, cleanup=note)
    info.append(note)
    if res["cycle"] is not None:
        info.append("planted near: " + res["cycle"]["planted_near"])
        info.append("planted far:  " + res["cycle"]["planted_far"])
    else:
        info.append("planted near: " + "".join("#" if c["planted"][0] else "." for c in res["contact"]))
        info.append("planted far:  " + "".join("#" if c["planted"][1] else "." for c in res["contact"]))
    preview = {"scan_frames": src_frames, "mirrored": mirrored, "width": W, "height": H, "clean": res["clean"],
               "src": res["src"], "contact": res["contact"], "ground": res["ground"], "order": order}
    return track, "\n".join(info), preview
