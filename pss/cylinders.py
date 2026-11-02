"""SCAIL pose video (z-buffered 3D cylinders) from a 2D skeleton track.

Logic copied unchanged from the workflow bundle's scripts/scail_pose_from_track.py; only the file I/O moved out.
Renders the pose video that drives SCAIL-Preview from your own OpenPose-18 track, without a driving video and without
a 3D pose estimator.

SCAIL-Preview was trained on 3D-cylinder renders of NLF 3D joints (zai-org/SCAIL-Pose; kijai port
ComfyUI-SCAIL-Pose NLFPoseExtract/nlf_render.py + render_3d/taichi_cylinder.py). This reproduces that renderer's look
(13 bones, ordered_colors_255 -> c/300+0.15, radius 21.5 mm, fov 55 deg pinhole, Blinn-Phong headlight
diffuse 0.3+0.7*n.l, white specular 0.5*spec^32, depth_factor 1-(z-near)/(far-znear) with near = clip z-min,
far = clip z-max + 6000 mm, black background) and replaces NLF with a lift of the OpenPose-18 track:

  - each 2D joint (u, v) is back-projected at its own depth Z, so the render projects EXACTLY onto the track;
  - Z = Z0 + dz, Z0 = f * mm_per_px (figure at the distance where H_px pixels = height_mm);
  - the near/far convention: OpenPose R side
    (2,3,4,8,9,10) = near = toward the camera (dz < 0), L side = far (dz > 0); a limb keeps its root's dz;
  - the root dz follows the torso turn: half-separation sqrt(S^2 - (d2d/2)^2) from the 2D shoulder (hip) distance,
    S = shoulder_mm (hip_mm), floored at min_sep_mm so the near side always stays in front;
  - head joint (SMPL 'head' -> OpenPose slot 0 in SCAIL-Pose) = ear + 0.35 (nose - ear), 0.01 H up; neck/head dz 0.
"""
import math
from types import SimpleNamespace
import numpy as np

# SCAIL-Pose (nlf_render.py) limb table, OpenPose-18 indices, and its ordered_colors_255
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0)]
COLORS_255 = [(255, 0, 0), (0, 255, 255), (255, 85, 0), (255, 170, 0), (0, 170, 255), (0, 85, 255), (180, 255, 0),
              (0, 255, 0), (0, 255, 85), (0, 0, 255), (85, 0, 255), (170, 0, 255), (150, 150, 150)]
COLORS = np.array([[c / 300 + 0.15 for c in rgb] for rgb in COLORS_255], np.float32)
NEAR = (2, 3, 4, 8, 9, 10)
FAR = (5, 6, 7, 11, 12, 13)


def load_track_data(kp):
    """kp = the loaded keypoints.json of a track (list of frames)"""
    P = np.array([np.array(f["people"][0]["pose_keypoints_2d"], float).reshape(-1, 3)[:18] for f in kp])
    return P, kp[0].get("canvas_width", 1024), kp[0].get("canvas_height", 1024)


def track_height(P):
    """nose-to-ankle height of the track in px (max over frames of nose y to the lower ankle), as depth_from_track"""
    return float(np.max(np.maximum(P[:, 10, 1], P[:, 13, 1]) - P[:, 0, 1]))


def lift(P, s, W, H, a):
    """2D track (canvas px) -> 3D joints (mm, camera frame, +Z forward, y down) on the SIZE x SIZE output canvas"""
    f = max(W, H) * s / (2 * math.tan(math.radians(a.fov) / 2))
    cx, cy = W * s / 2, H * s / 2 - a.crop_off
    Hpx = track_height(P) * s
    mmpp = a.height_mm / Hpx
    Z0 = f * mmpp
    J = np.zeros((len(P), 18, 3), np.float32)
    info = []
    for i, p in enumerate(P):
        uv = p[:, :2] * s - np.array([0, a.crop_off])
        def sep(j1, j2, S):
            d = np.linalg.norm(uv[j1] - uv[j2]) * mmpp
            return max(math.sqrt(max(S * S - (d / 2) ** 2, 0.0)), a.min_sep_mm), d
        zs, ds = sep(2, 5, a.shoulder_mm)
        zh, dh = sep(8, 11, a.hip_mm)
        dz = np.zeros(18)
        for j in (2, 3, 4): dz[j] = -zs
        for j in (5, 6, 7): dz[j] = +zs
        for j in (8, 9, 10): dz[j] = -zh
        for j in (11, 12, 13): dz[j] = +zh
        ears = [uv[j] for j in (16, 17) if p[j, 2] > 0]
        ear = np.mean(ears, 0) if ears else uv[0] - np.array([0.05 * Hpx, 0])
        head = ear + (uv[0] - ear) * 0.35 + np.array([0, -0.01 * Hpx])
        uvh = uv.copy()
        uvh[0] = head
        Z = Z0 + dz
        J[i, :, 0] = (uvh[:, 0] - cx) * Z / f
        J[i, :, 1] = (uvh[:, 1] - cy) * Z / f
        J[i, :, 2] = Z
        info.append({"shoulder_2d_mm": round(ds, 1), "shoulder_half_sep_z_mm": round(zs, 1), "hip_2d_mm": round(dh, 1),
                     "hip_half_sep_z_mm": round(zh, 1)})
    return J, dict(f=f, cx=cx, cy=cy, Z0=Z0, mm_per_px=mmpp, H_px=Hpx), info


def ray_capsule_t(rd, A, B, r):
    """closed-form ray (origin 0, unit dirs rd (K,3)) vs capsules A->B radius r: (K, M) nearest t, inf on miss"""
    ba = B - A
    L = np.linalg.norm(ba, axis=1).clip(1e-6)
    n = ba / L[:, None]
    dn = rd @ n.T
    dA = rd @ A.T
    dB = rd @ B.T
    An = (A * n).sum(1)
    A2 = (A * A).sum(1)
    B2 = (B * B).sum(1)
    r2 = r * r
    inf = np.inf
    a_c = 1 - dn * dn
    b_c = -2 * (dA - dn * An)
    c_c = A2 - An * An - r2
    disc = b_c * b_c - 4 * a_c * c_c
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-b_c - np.sqrt(disc.clip(0))) / (2 * a_c.clip(1e-9))
        s_ = t * dn - An
        t_c = np.where((disc >= 0) & (a_c > 1e-7) & (t > 1e-6) & (s_ >= 0) & (s_ <= L), t, inf)
        da = dA * dA - (A2 - r2)
        ta = dA - np.sqrt(da.clip(0))
        sa = ta * dn - An
        t_a = np.where((da >= 0) & (ta > 1e-6) & (sa <= 0), ta, inf)
        db = dB * dB - (B2 - r2)
        tb = dB - np.sqrt(db.clip(0))
        sb = tb * dn - An
        t_b = np.where((db >= 0) & (tb > 1e-6) & (sb >= L), tb, inf)
    return np.minimum(np.minimum(t_c, t_a), t_b)


def render(J, cam, Wo, Ho, r_mm, near, far):
    A = np.array([J[a] for a, b in LIMBS], np.float64)
    B = np.array([J[b] for a, b in LIMBS], np.float64)
    ys, xs = np.mgrid[0:Ho, 0:Wo].astype(np.float64)
    rd = np.stack([(xs.ravel() + 0.5 - cam["cx"]) / cam["f"], (ys.ravel() + 0.5 - cam["cy"]) / cam["f"], np.ones(xs.size)], 1)
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    t = ray_capsule_t(rd, A, B, np.full(len(LIMBS), r_mm))
    m = t.argmin(1)
    tm = t[np.arange(len(m)), m]
    hit = np.isfinite(tm)
    out = np.zeros((xs.size, 3), np.float32)
    rh, th, mh = rd[hit], tm[hit], m[hit]
    p = rh * th[:, None]
    n_ = (B - A)[mh]
    Ln = np.linalg.norm(n_, axis=1).clip(1e-6)
    n_ = n_ / Ln[:, None]
    proj = np.clip(((p - A[mh]) * n_).sum(1), 0, Ln)
    nrm = p - (A[mh] + proj[:, None] * n_)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-8)
    light = np.array([0, 0, 1.0])
    diff = np.clip(nrm @ -light, 0, None)
    half = -rh + -light
    half /= np.linalg.norm(half, axis=1, keepdims=True).clip(1e-8)
    spec = np.clip((nrm * half).sum(1), 0, None) ** 32
    depth = np.clip(1 - (p[:, 2] - near) / (far - 0.1), 0, 1)
    out[hit] = COLORS[mh] * ((0.3 + 0.7 * diff) * depth)[:, None] + (0.5 * spec * depth)[:, None]
    return (np.clip(out, 0, 1).reshape(Ho, Wo, 3) * 255 + 0.5).astype(np.uint8)


def render_track(keypoints, track_name="track", size=768, crop_h=0, crop_off=None, height_mm=1550.0,
                 shoulder_mm=200.0, hip_mm=120.0, min_sep_mm=60.0, radius_mm=21.5, fov=55.0, progress=None):
    """keypoints = the loaded keypoints.json of a (fitted) track. crop_off None = centred (the script's default).
    Returns (list of HxWx3 uint8 frames, params dict as the script's params.json, joints3d dict).
    The body is scail_pose_from_track.py's main() with the file writes replaced by returning the same objects."""
    a = SimpleNamespace(size=size, crop_h=crop_h, crop_off=crop_off, height_mm=height_mm, shoulder_mm=shoulder_mm,
                        hip_mm=hip_mm, min_sep_mm=min_sep_mm, radius_mm=radius_mm, fov=fov)
    P, W, H = load_track_data(keypoints)
    s = a.size / W
    Ho = a.crop_h or int(round(H * s))
    a.crop_off = a.crop_off if a.crop_off is not None else (int(round(H * s)) - Ho) // 2
    J, cam, info = lift(P, s, W, H, a)
    z = J[..., 2]
    near = max(float(z.min()), 0.1)
    far = min(float(z.max()) + 6000, 20000)
    frames = []
    for i in range(len(J)):
        frames.append(render(J[i], cam, a.size, Ho, a.radius_mm, near, far))
        if progress is not None:
            progress(i + 1, len(J))
    joints = {"joints3d_mm": J.round(1).tolist(), "per_frame": info}
    params = {"track": track_name, "canvas": [W, H], "frames": len(J), "out_size": [a.size, Ho], "crop_off": a.crop_off,
              "camera": {k: round(v, 3) for k, v in cam.items()}, "fov_deg": a.fov, "height_mm": a.height_mm,
              "shoulder_mm": a.shoulder_mm, "hip_mm": a.hip_mm, "min_sep_mm": a.min_sep_mm, "radius_mm": a.radius_mm,
              "depth_near_far_mm": [round(near, 1), round(far, 1)],
              "renderer": "SCAIL-Pose taichi_cylinder.py look re-implemented (closed-form ray-capsule, as ComfyUI)",
              "bones": "13: neck-shoulders, arms, neck-hips, legs, neck-head (SCAIL-Pose draw_seq without the unset face bones)",
              "colors": "SCAIL-Pose ordered_colors_255 -> c/300 + 0.15", "convention": "OpenPose R (2,3,4,8,9,10) = near (toward the camera)",
              "tool": "Render SCAIL Cylinders (posed-sprite-sheet; logic of scail_pose_from_track.py)"}
    return frames, params, joints
