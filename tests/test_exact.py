"""Exactness gate: run the four nodes' functions directly (no server) on the workflow bundle's inputs with the README's
per-move settings and compare with the bundle's files.

usage: python tests/test_exact.py BUNDLE_DIR OUT_DIR [--moves walk short_uppercut heavy_uppercut]
  BUNDLE_DIR = the extracted "Posed Sprite Sheet Generator V2" folder (holds inputs/, scripts/ ...)
  OUT_DIR    = new folder (must not exist): node outputs (pose PNGs, padded stills, fitted keypoints) + report.json
Exit code 0 only if every comparison is exact.
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch
from PIL import Image, ImageOps

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the README's per-move settings (fit_track.py / scail_pose_from_track.py / prep_ref_pad.py commands)
MOVES = {
    "short_uppercut": dict(fit=dict(canvas=1024, hip_x=347.0, fixed_shift=False, pad_top=0, scale_frame=0),
                           render=dict(crop_h=0, crop_off=-1),
                           pad=dict(still="short_uppercut_side_raw.png", ref="short_uppercut_pad.png",
                                    cx=289.0, y0=40, y1=691, keep_size=False)),
    "walk": dict(fit=dict(canvas=1024, hip_x=347.0, fixed_shift=True, pad_top=0, scale_frame=0),
                 render=dict(crop_h=0, crop_off=-1),
                 pad=dict(still="walk_side_raw.png", ref="walk_pad.png", cx=249.0, y0=28, y1=679, keep_size=False)),
    "heavy_uppercut": dict(fit=dict(canvas=1240, hip_x=420.0, fixed_shift=False, pad_top=269, scale_frame=0),
                           render=dict(crop_h=768, crop_off=0),
                           pad=dict(still="short_uppercut_side_raw.png", ref="heavy_uppercut_pad.png",
                                    cx=284.0, y0=200, y1=737, keep_size=True)),
}
RENDER_DEFAULTS = dict(size=768, height_mm=1550.0, shoulder_mm=200.0, hip_mm=120.0, min_sep_mm=60.0, radius_mm=21.5,
                       fov=55.0)


def import_pack():
    spec = importlib.util.spec_from_file_location("posed_sprite_sheet", os.path.join(PACK, "__init__.py"),
                                                  submodule_search_locations=[PACK])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["posed_sprite_sheet"] = mod
    spec.loader.exec_module(mod)
    return sys.modules["posed_sprite_sheet.nodes"]


def load_image_like_comfy(path):
    """what ComfyUI's LoadImage node does for an RGB png: exif_transpose, convert RGB, float32 / 255, batch of 1"""
    i = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return torch.from_numpy(np.array(i).astype(np.float32) / 255.0)[None]


def cmp_frames(ours, ref_paths):
    """ours: (N,H,W,3) uint8; ref_paths: bundle pngs. Returns counts and max abs difference."""
    same, maxd, shape_bad = 0, 0, 0
    for a, p in zip(ours, ref_paths):
        b = np.asarray(Image.open(p).convert("RGB"))
        if a.shape != b.shape:
            shape_bad += 1
            continue
        d = int(np.abs(a.astype(int) - b.astype(int)).max())
        maxd = max(maxd, d)
        same += d == 0
    return dict(identical=same, total=len(ref_paths), count_match=len(ours) == len(ref_paths), max_diff=maxd,
                shape_mismatch=shape_bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("out")
    ap.add_argument("--moves", nargs="+", default=list(MOVES))
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists: new dir per run")
    os.makedirs(a.out)
    N = import_pack()
    B = os.path.abspath(os.path.join(a.bundle, "inputs"))
    a.out = os.path.abspath(a.out)
    os.chdir(B)   # 1.2.2: node paths must be relative (outside ComfyUI they are relative to the working directory)
    cop_kps = os.path.join(B, "scail_cop", "cop", "cop_keypoints.json")
    with open(cop_kps) as f:
        cop_pose_keypoint = json.load(f)              # same structure as comfyui_controlnet_aux's POSE_KEYPOINT
    report = {"numpy": np.__version__, "torch": torch.__version__, "pillow": Image.__version__, "moves": {}}
    ok = True
    for move in a.moves:
        cfg = MOVES[move]
        r = {}
        t0 = time.time()
        D = os.path.join(a.out, move)
        os.makedirs(D)
        # 1. Load Skeleton Track (source) -> Fit Track to Character (json file input, and POSE_KEYPOINT input)
        src, n_src, info = N.PSS_LoadSkeletonTrack().load(os.path.join("tracks", move, "source"))
        fitted, check, fit_info = N.PSS_FitTrackToCharacter().fit(src, os.path.relpath(cop_kps, B), **cfg["fit"])
        fitted_pk, _, _ = N.PSS_FitTrackToCharacter().fit(src, "", char_keypoints=cop_pose_keypoint, **cfg["fit"])
        ref_kp_path = os.path.join(B, "tracks", move, "fitted", "keypoints.json")
        ref_bytes = open(ref_kp_path, "rb").read()
        ours_bytes = json.dumps(fitted["keypoints"]).encode()
        with open(os.path.join(D, "fitted_keypoints.json"), "wb") as f:
            f.write(ours_bytes)
        Image.fromarray(N.image_to_uint8(check)[0]).save(os.path.join(D, "fit_check.png"))
        r["fit"] = {"json_equal": fitted["keypoints"] == json.loads(ref_bytes),
                    "bytes_identical": ours_bytes == ref_bytes,
                    "pose_keypoint_input_equal": fitted_pk["keypoints"] == fitted["keypoints"],
                    "frames": len(fitted["keypoints"]), "info": fit_info.splitlines()[0]}
        # 2. Render SCAIL Cylinders on the node-fitted track (the whole chain)
        img, params = N.PSS_RenderSCAILCylinders().render(fitted, **RENDER_DEFAULTS, **cfg["render"])
        ours = N.image_to_uint8(img)
        os.makedirs(os.path.join(D, "pose"))
        for i, fr in enumerate(ours):
            Image.fromarray(fr).save(os.path.join(D, "pose", f"pose_{i:03d}.png"))
        with open(os.path.join(D, "params.json"), "w") as f:
            f.write(params)
        ref_dir = os.path.join(B, "scail_cop", "poses", move)
        refs = sorted(os.path.join(ref_dir, p) for p in os.listdir(ref_dir) if p.endswith(".png"))
        r["pose"] = cmp_frames(ours, refs)
        r["pose"]["tensor_dtype"] = str(img.dtype)
        r["pose"]["tensor_shape"] = list(img.shape)
        # the same frames re-read from the PNGs we saved (the SaveImage path)
        reread = np.stack([np.asarray(Image.open(os.path.join(D, "pose", f"pose_{i:03d}.png"))) for i in range(len(ours))])
        r["pose_png_reread"] = cmp_frames(reread, refs)
        # also render from the bundle's own fitted track folder (Load Skeleton Track -> Render)
        bt, _, _ = N.PSS_LoadSkeletonTrack().load(os.path.join("tracks", move, "fitted"))
        img_b, _ = N.PSS_RenderSCAILCylinders().render(bt, **RENDER_DEFAULTS, **cfg["render"])
        r["pose_from_bundle_track"] = cmp_frames(N.image_to_uint8(img_b), refs)
        # 3. Pad Still to Pose (manual, the README numbers) and the optional auto mode from pose frame 0
        p = cfg["pad"]
        still = load_image_like_comfy(os.path.join(B, "scail_cop", "refs", p["still"]))
        padded, pinfo = N.PSS_PadStillToPose().pad(still, p["cx"], p["y0"], p["y1"], p["keep_size"], "manual")
        pu = N.image_to_uint8(padded)
        Image.fromarray(pu[0]).save(os.path.join(D, "padded.png"))
        ref_pad = os.path.join(B, "scail_cop", "refs", p["ref"])
        r["pad"] = cmp_frames(pu, [ref_pad])
        r["pad"]["info"] = pinfo
        padded_a, ainfo = N.PSS_PadStillToPose().pad(still, 0.0, 0, 0, p["keep_size"], "auto from pose frame 0",
                                                     pose_frames=img)
        r["pad_auto"] = cmp_frames(N.image_to_uint8(padded_a), [ref_pad])
        r["pad_auto"]["info"] = ainfo
        r["seconds"] = round(time.time() - t0, 1)
        exact = (r["fit"]["json_equal"] and r["fit"]["pose_keypoint_input_equal"]
                 and all(r[k]["identical"] == r[k]["total"] and r[k]["count_match"]
                         for k in ("pose", "pose_png_reread", "pose_from_bundle_track", "pad")))
        r["exact"] = exact
        ok &= exact
        report["moves"][move] = r
        print(move, json.dumps(r, indent=1), flush=True)
    report["all_exact"] = ok
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("ALL EXACT" if ok else "NOT EXACT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
