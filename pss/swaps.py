"""Repair a pose scan (logic of the workflow's fix_lr_swaps.py; file I/O moved out; one fix: of the two equal-cost
labellings the one closer to DWPose's own labels is kept, see do()).

1. Left/right flips: in a side view DWPose swaps the labels of whole limbs for a few frames when the legs cross. Per
   limb set (legs: hip/knee/ankle + heel/toes; arms: shoulder/elbow/wrist) a Viterbi pass over "swap this frame or
   not" minimises the second difference (acceleration) of both limbs' points: a real limb moves smoothly, a label flip
   is a jump.
2. Lost points (x <= 1 or score < 0.15) are interpolated in time, then a 3-frame median removes one-frame spikes.
3. A heel/toe further than 1.3x its median ankle distance is pulled back to that length (direction kept).

Frames without a person are filled from the nearest frame with one (score 0) first, so pass 2 interpolates them.
"""
import copy
import itertools

import numpy as np

LEG = [(8, 11), (9, 12), (10, 13)]
ARM = [(2, 5), (3, 6), (4, 7)]
FEET_PAIRS = [("right_big_toe", "left_big_toe"), ("right_small_toe", "left_small_toe"), ("right_heel", "left_heel")]


def _fill_empty(fr, feet_order):
    have = [t for t, f in enumerate(fr) if f["body"] is not None]
    if not have:
        raise ValueError("no frame of the scan has a person in it")
    n = 0
    for t, f in enumerate(fr):
        if f["body"] is None:
            src = fr[min(have, key=lambda s: abs(s - t))]
            f["body"] = [[p[0], p[1], 0.0] for p in src["body"]]
            f["feet"] = {k: [src["feet"][k][0], src["feet"][k][1], 0.0] for k in feet_order}
            n += 1
    return n


def _solve(R, L, flip_cost=0.0):
    T = len(R)

    def pos(t, s):
        return (L[t], R[t]) if s else (R[t], L[t])    # (right-label, left-label) after optional swap

    def acc(a_, b_, c_):
        return sum(np.abs(x[0] - 2 * x[1] + x[2]).sum() for x in zip(a_, b_, c_))
    cost = {(s0, s1): flip_cost * (s0 + s1) for s0 in (0, 1) for s1 in (0, 1)}
    back = []
    for t in range(2, T):
        new, bp = {}, {}
        for s1, s2 in itertools.product((0, 1), repeat=2):
            best = None
            for s0 in (0, 1):
                c = cost[(s0, s1)] + acc(pos(t - 2, s0), pos(t - 1, s1), pos(t, s2)) + flip_cost * s2
                if best is None or c < best[0]:
                    best = (c, s0)
            new[(s1, s2)] = best[0]
            bp[(s1, s2)] = best[1]
        cost = new
        back.append(bp)
    st = min(cost, key=cost.get)
    s = [st[0], st[1]]
    for bp in reversed(back):
        s0 = bp[(s[0], s[1])]
        s.insert(0, s0)
    return s[-T:]


def repair(scan, fix_swaps=True, interpolate=True, clamp_feet=True, flip_cost=-1.0):
    """returns (new scan, report lines). flip_cost: px of smoothness a swapped frame must buy (the workflow script's
    --flip-cost, which it never applied); -1 = auto, 0.5% of the figure height. Without it, a stretch between two
    frames where both limbs coincide (arms passing each other) can be relabelled for free, and float noise decides."""
    j = copy.deepcopy(scan)
    fr = j["frames"]
    order = j["feet_order"]
    report = []
    ne = _fill_empty(fr, order)
    if ne:
        report.append(f"{ne} frames without a person filled from the nearest frame (score 0)")
    if len(fr) < 3:
        return j, report + ["fewer than 3 frames: nothing repaired"]
    notes = []
    if flip_cost < 0:
        hs = []
        for f in fr:
            ys = [p[1] for p in f["body"] if p[2] > 0.3] + [f["feet"][k][1] for k in order if f["feet"][k][2] > 0.3]
            if len(ys) > 4:
                hs.append(max(ys) - min(ys))
        flip_cost = 0.005 * float(np.median(hs)) if hs else 0.0
    if fix_swaps:
        report.append(f"flip cost {flip_cost:.2f} px per swapped frame")
        def arr(grp, feet):
            R = np.array([[f["body"][r][:2] for r, l in grp] + ([f["feet"][r][:2] for r, l in feet] if feet else [])
                          for f in fr], float)
            L = np.array([[f["body"][l][:2] for r, l in grp] + ([f["feet"][l][:2] for r, l in feet] if feet else [])
                          for f in fr], float)
            return R, L

        def do(grp, feet, name):
            R, L = arr(grp, feet)
            s = _solve(R, L, flip_cost)
            # the smoothness cost is the same for a labelling and its complement (swap every frame), so the Viterbi
            # fixes WHERE the labels flip but not which way round they are: keep the one that agrees with DWPose's own
            # labels on most frames (DWPose flips for a few frames). Without this, float noise picks one at random.
            if sum(s) * 2 > len(s):
                s = [1 - v for v in s]
            n = 0
            for t, f in enumerate(fr):
                if s[t] and f["body"] is not None:
                    for r, l in grp:
                        f["body"][r], f["body"][l] = f["body"][l], f["body"][r]
                    if feet:
                        for r, l in feet:
                            f["feet"][r], f["feet"][l] = f["feet"][l], f["feet"][r]
                    n += 1
            report.append(f"{name} swapped frames: {[t for t in range(len(fr)) if s[t]]}")
            return n
        n = do(LEG, FEET_PAIRS, "leg") + do(ARM, None, "arm")
        notes.append(f"fix_lr_swaps (viterbi): {n} limb swaps")

    if interpolate:
        def series(get, set_):
            T = len(fr)
            v = np.array([get(f) for f in fr], float)
            bad = (v[:, 0] <= 1) | (v[:, 2] < 0.15)
            if bad.any() and (~bad).sum() > 2:
                idx = np.arange(T)
                for c in (0, 1):
                    v[bad, c] = np.interp(idx[bad], idx[~bad], v[~bad, c])
                v[bad, 2] = 0.5
            m = v.copy()
            for t in range(1, T - 1):
                m[t, :2] = np.median(v[t - 1:t + 2, :2], axis=0)
            for t, f in enumerate(fr):
                set_(f, m[t].tolist())
            return int(bad.sum())
        nb = 0
        for i in range(18):
            nb += series(lambda f, i=i: f["body"][i], lambda f, v, i=i: f["body"].__setitem__(i, v))
        for k in order:
            nb += series(lambda f, k=k: f["feet"][k], lambda f, v, k=k: f["feet"].__setitem__(k, v))
        report.append(f"interpolated points: {nb}")
        notes.append(f"{nb} lost points interpolated, 3-frame median")

    if clamp_feet:
        nf = 0
        for side, ank in (("right", 10), ("left", 13)):
            for nm in ("big_toe", "small_toe", "heel"):
                k = f"{side}_{nm}"
                D = np.array([np.subtract(f["feet"][k][:2], f["body"][ank][:2]) for f in fr], float)
                Ln = np.hypot(D[:, 0], D[:, 1])
                med = np.median(Ln)
                for t, f in enumerate(fr):
                    if Ln[t] > 1.3 * med:
                        f["feet"][k][:2] = (np.array(f["body"][ank][:2]) + D[t] * 1.3 * med / Ln[t]).tolist()
                        nf += 1
        report.append(f"foot points shortened: {nf}")
        notes.append(f"{nf} foot points clamped")
    j["note"] = j.get("note", "") + "".join(" | " + n for n in notes)
    return j, report
