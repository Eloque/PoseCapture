"""ComfyUI nodes: a video of a person -> pose scan (DWPose with the feet) -> repaired scan -> skeleton track, which then
goes into Fit Track to Character and Render SCAIL Cylinders.

The pose scan needs comfyui_controlnet_aux (its DWPose models: yolox_l + dw-ll_ucoco_384); everything after it is
numpy + Pillow on the CPU.
"""
import json
import os

import numpy as np

from .nodes import CATEGORY, TRACK_TYPE, frames_to_image, image_to_uint8, inside, plain_relative, resolve, track_info
from .pss import preview as pss_prev
from .pss import scan as pss_scan
from .pss import swaps as pss_swaps
from .pss import track as pss_track

SCAN_TYPE = "PSS_POSE_SCAN"
DETECTORS = ["yolox_l.onnx", "yolox_l.torchscript.pt"]
ESTIMATORS = ["dw-ll_ucoco_384_bs5.torchscript.pt", "dw-ll_ucoco_384.onnx"]


def output_dir():
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return os.path.join(os.getcwd(), "output")


def next_dir(base, name):
    """base/name_vN with the first N that does not exist (never overwrite)"""
    n = 1
    while os.path.exists(os.path.join(base, f"{name}_v{n}")):
        n += 1
    d = os.path.join(base, f"{name}_v{n}")
    os.makedirs(d)
    return d


def save_dir(folder, name):
    """a new output/<folder>/<name>_vN folder; folder must stay inside the ComfyUI output folder (by name and after
    resolving symlinks), name is a plain name"""
    out = output_dir()
    try:
        folder = plain_relative(folder)
    except ValueError:
        raise ValueError(f"PoseCapture: folder must be relative to ComfyUI/output/, not {folder!r}")
    name = name.strip()
    base = os.path.join(out, folder)
    if not inside(out, base):
        raise ValueError(f"PoseCapture: folder must stay inside ComfyUI/output/, not {folder!r}")
    if (not name or name in (".", "..") or os.path.basename(name) != name or "/" in name or "\\" in name
            or "\0" in name):
        raise ValueError(f"PoseCapture: name must be a plain name, not {name!r}")
    d = next_dir(base, name)
    if not inside(out, d):     # checked again on the folder that now exists
        raise ValueError(f"PoseCapture: {d!r} is outside ComfyUI/output/")
    return d


def read_video(path, start=0, frames=0, every=1, crop=None, wanted=None):
    """yields (source frame number, BGR uint8 crop). wanted = a set of frame numbers (overrides start/frames/every)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open the video '{path}'")
    i, n = 0, 0
    last = max(wanted) if wanted else None
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if wanted is not None:
                if i > last:
                    break
                if i in wanted:
                    yield i, pss_scan.apply_crop(img, crop)
            elif i >= start and (i - start) % every == 0:
                if frames and n >= frames:
                    break
                yield i, pss_scan.apply_crop(img, crop)
                n += 1
            i += 1
    finally:
        cap.release()


def video_fps(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps


def load_dwpose(bbox_detector, pose_estimator):
    try:
        from custom_controlnet_aux.dwpose import DwposeDetector
    except ImportError as e:
        raise ImportError("Scan Video Poses needs the comfyui_controlnet_aux custom nodes (for DWPose); install them "
                          "with the ComfyUI Manager") from e
    import comfy.model_management as mm
    yolo_repo = "yzd-v/DWPose" if bbox_detector == "yolox_l.onnx" else "hr16/yolox-onnx"
    if pose_estimator == "dw-ll_ucoco_384.onnx":
        pose_repo = "yzd-v/DWPose"
    else:
        pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
    det = DwposeDetector.from_pretrained(pose_repo, yolo_repo, det_filename=bbox_detector, pose_filename=pose_estimator,
                                         torchscript_device=mm.get_torch_device())
    return det.dw_pose_estimation


class PSS_ScanVideoPoses:
    DESCRIPTION = ("Runs DWPose whole-body on every frame of a video and keeps the 18 body points AND the six foot "
                   "points (heel, big toe, small toe of each foot), which the usual DWPose keypoints drop: the feet "
                   "give the ground contact. Connect the frames of a Load Video node to images (or, for a long video, "
                   "leave images unconnected and give a video file path: it is then read frame by frame without "
                   "holding the whole clip in memory). crop = x,y,w,h of the person's part of the frame (empty = the "
                   "whole frame). About 1 s per frame on a CPU. Needs comfyui_controlnet_aux.")
    CATEGORY = CATEGORY
    RETURN_TYPES = (SCAN_TYPE, "IMAGE", "STRING")
    RETURN_NAMES = ("scan", "overlay", "info")
    FUNCTION = "scan"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "crop": ("STRING", {"default": "", "tooltip": "x,y,w,h in pixels of the frames; empty = whole frame"}),
            "start": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "first frame"}),
            "frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "frames to scan, 0 = to the end"}),
            "every": ("INT", {"default": 1, "min": 1, "max": 100, "tooltip": "take every n-th frame"}),
            "bbox_detector": (DETECTORS, {"default": DETECTORS[0]}),
            "pose_estimator": (ESTIMATORS, {"default": ESTIMATORS[0]}),
            "overlay_height": ("INT", {"default": 720, "min": 0, "max": 4096,
                                       "tooltip": "height of the overlay frames (0 = scan size)"}),
        }, "optional": {
            "images": ("IMAGE", {"tooltip": "the video frames (from Load Video)"}),
            "video": ("STRING", {"default": "", "tooltip": "used only when images is not connected: a video file, "
                                                           "relative to ComfyUI/input"}),
        }}

    def scan(self, crop, start, frames, every, bbox_detector, pose_estimator, overlay_height, images=None, video=""):
        import comfy.utils
        model = load_dwpose(bbox_detector, pose_estimator)
        c = pss_scan.parse_crop(crop)
        if images is not None:
            T = images.shape[0]
            end = T if frames <= 0 else min(T, start + frames * every)
            nums = list(range(start, end, every))
            bgr = [pss_scan.apply_crop(image_to_uint8(images[i:i + 1])[0, :, :, 2::-1], c) for i in nums]
            path, fps = "", 0.0
        else:
            if not video:
                raise ValueError("connect the video frames to images, or give a video file path")
            path = resolve(video)
            fps = video_fps(path)
            pairs = list(read_video(path, start, frames, every, c))
            nums = [p[0] for p in pairs]
            bgr = [p[1] for p in pairs]
        if not bgr:
            raise ValueError("no frames to scan")
        pbar = comfy.utils.ProgressBar(len(bgr))
        recs = pss_scan.scan_frames(bgr, model, nums, progress=lambda k: pbar.update_absolute(k, len(bgr)))
        h, w = bgr[0].shape[:2]
        scan = pss_scan.new_scan(recs, video=os.path.basename(path), crop=c, start=start, every=every,
                                 fps=fps / every if fps else 0.0, size=(w, h))
        scan["_source"] = path        # full path for the preview, not written by Save Pose Scan
        ov = [pss_prev.scan_overlay(np.ascontiguousarray(b[:, :, ::-1]), r, scan["feet_order"]) for b, r in zip(bgr, recs)]
        ov = pss_prev.fit_height(ov, overlay_height)
        return (scan, frames_to_image(ov), pss_scan.summary(scan))


class PSS_LoadPoseScan:
    DESCRIPTION = "Loads a pose scan saved by Save Pose Scan (keypoints_feet.json), relative to ComfyUI/input (else ComfyUI/output)."
    CATEGORY = CATEGORY
    RETURN_TYPES = (SCAN_TYPE, "STRING")
    RETURN_NAMES = ("scan", "info")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "pss_scans/walk_v1/keypoints_feet.json"}),
                             "video": ("STRING", {"default": "", "tooltip": "the scanned video, for the previews "
                                                                            "(empty = the name stored in the scan)"})}}

    @classmethod
    def IS_CHANGED(cls, path, video):
        p = resolve(path)
        return f"{os.stat(p).st_mtime_ns}:{video}" if os.path.isfile(p) else path

    def load(self, path, video):
        with open(resolve(path)) as f:
            scan = json.load(f)
        v = video or scan.get("video") or ""
        if v:
            scan["_source"] = resolve(v)
        return (scan, pss_scan.summary(scan))


class PSS_SavePoseScan:
    DESCRIPTION = ("Saves a pose scan as keypoints_feet.json in ComfyUI/output/<folder>/<name>_vN/ (a new folder every "
                   "time, never overwriting), so the slow scan does not have to run again.")
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scan": (SCAN_TYPE,), "folder": ("STRING", {"default": "pss_scans"}),
                             "name": ("STRING", {"default": "scan"})}}

    def save(self, scan, folder, name):
        d = save_dir(folder, name)
        p = os.path.join(d, "keypoints_feet.json")
        with open(p, "w") as f:
            json.dump({k: v for k, v in scan.items() if not k.startswith("_")}, f)
        return {"ui": {"text": [p]}, "result": (p,)}


class PSS_RepairPoseScan:
    DESCRIPTION = ("Repairs a pose scan: (1) left/right label flips (in a side view DWPose swaps whole limbs for a few "
                   "frames when the legs cross; a Viterbi pass picks the labelling with the smoothest motion), (2) lost "
                   "points interpolated in time + a 3-frame median against one-frame spikes, (3) heel/toe points too "
                   "far from the ankle pulled back. Frames without a person are filled first.")
    CATEGORY = CATEGORY
    RETURN_TYPES = (SCAN_TYPE, "STRING")
    RETURN_NAMES = ("scan", "info")
    FUNCTION = "repair"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scan": (SCAN_TYPE,), "fix_swaps": ("BOOLEAN", {"default": True}),
                             "interpolate": ("BOOLEAN", {"default": True}),
                             "clamp_feet": ("BOOLEAN", {"default": True}),
                             "flip_cost": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.5,
                                                     "tooltip": "px of smoothness a swapped frame must gain (stops "
                                                                "free relabelling where two limbs cross); -1 = 0.5% "
                                                                "of the figure height"})}}

    def repair(self, scan, fix_swaps, interpolate, clamp_feet, flip_cost=-1.0):
        out, report = pss_swaps.repair(scan, fix_swaps, interpolate, clamp_feet, flip_cost)
        return (out, "\n".join(report))


class PSS_TrackFromPoseScan:
    DESCRIPTION = ("Turns a pose scan into a skeleton track (input of Fit Track to Character): mirrored to face right if "
                   "needed, near side labelled R, both feet on one ground line, planted feet locked. mode 'walk cycle': "
                   "one cycle start..start+period-1 closed into a seamless loop, planted feet moving back at one "
                   "treadmill speed (a walk across the frame is put on a treadmill first); period 0 = found "
                   "automatically; frames 0 = just over two cycles. mode 'one-shot': the frames start..start+frames-1 "
                   "as they are (punch, kick, step-in), planted feet held still. The preview draws the track over the "
                   "video: R = near = orange, L = far = blue, green = planted.")
    CATEGORY = CATEGORY
    RETURN_TYPES = (TRACK_TYPE, "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("track", "preview", "frames", "info")
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scan": (SCAN_TYPE,),
            "mode": (["walk cycle", "one-shot"], {"default": "walk cycle"}),
            "start": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "first frame, counted in the scan"}),
            "frames": ("INT", {"default": 0, "min": 0, "max": 100000,
                               "tooltip": "track length (4n+1 for SCAIL); walk: 0 = just over two cycles; one-shot: 0 = to the end"}),
            "period": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "walk cycle length in frames, 0 = auto"}),
            "face": (["auto", "right", "left"], {"default": "auto", "tooltip": "which way the person faces in the video"}),
            "near_side": (["auto", "right", "left"], {"default": "auto",
                                                     "tooltip": "DWPose's label of the side toward the camera"}),
            "remove_travel": (["auto", "on", "off"], {"default": "auto",
                                                     "tooltip": "walk cycle: subtract the body's travel (a walk across the frame)"}),
            "far_drop": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.5,
                                   "tooltip": "px the far foot moves down onto the ground line; -1 = measured"}),
            "ground_tol": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5,
                                     "tooltip": "px above the ground that still counts as down; 0 = 2.3% of the figure"}),
            "swing_tol": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5,
                                    "tooltip": "px/frame of forward motion that still counts as planted; 0 = 0.76% of the figure"}),
            "smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.25,
                                 "tooltip": "gaussian smoothing in time (frames) against DWPose jitter; 0 = off"}),
            "preview_height": ("INT", {"default": 720, "min": 0, "max": 4096}),
        }, "optional": {"images": ("IMAGE", {"tooltip": "the scanned video frames (the same Load Video output as the "
                                                        "scan), drawn behind the preview"})}}

    def make(self, scan, mode, start, frames, period, face, near_side, remove_travel, far_drop, ground_tol, swing_tol,
             smooth, preview_height, images=None):
        track, info, pv = pss_track.make_track(scan, mode, start, frames, period, face, near_side, remove_travel,
                                               far_drop, ground_tol, swing_tol, smooth)
        track["name"] = os.path.splitext(scan.get("video") or "video")[0] + ("_walk" if mode == "walk cycle" else "_move")
        frames_out = preview_frames(scan, pv, preview_height, images)
        return (track, frames_to_image(frames_out), len(track["keypoints"]), info + "\n" + track_info(track))


def preview_frames(scan, pv, max_h, images=None):
    W, H = pv["width"], pv["height"]
    src_path = scan.get("_source") or ""
    wanted = sorted(set(pv["scan_frames"]))
    imgs = {}
    crop = pss_scan.parse_crop(scan.get("crop") or "")
    if images is not None:
        for i in wanted:
            n = scan["frames"][i]["frame"]
            if n < images.shape[0]:
                rgb = np.ascontiguousarray(pss_scan.apply_crop(image_to_uint8(images[n:n + 1])[0, :, :, :3], crop))
                if rgb.shape[:2] == (H, W):
                    imgs[i] = rgb[:, ::-1] if pv["mirrored"] else rgb
    elif src_path and os.path.isfile(src_path):
        nums = [scan["frames"][i]["frame"] for i in wanted]
        by_num = dict(read_video(src_path, crop=crop, wanted=set(nums)))
        for i, n in zip(wanted, nums):
            if n in by_num:
                rgb = np.ascontiguousarray(by_num[n][:, :, ::-1])
                imgs[i] = rgb[:, ::-1] if pv["mirrored"] else rgb
    out = []
    for t, (si, c) in enumerate(zip(pv["scan_frames"], pv["src"])):
        bg = imgs.get(si, np.zeros((H, W, 3), np.uint8))
        out.append(pss_prev.track_overlay(bg, pv["clean"][c], pv["order"], pv["contact"][t], pv["ground"]))
    return pss_prev.fit_height(out, max_h)


class PSS_SaveSkeletonTrack:
    DESCRIPTION = ("Saves a skeleton track as a track folder (keypoints.json, feet.json, contact.json, cycle.json, "
                   "fit.json as present) in ComfyUI/output/<folder>/<name>_vN/ (a new folder every time). Load it again "
                   "with Load Skeleton Track by the folder this node prints (e.g. pss_tracks/kick_v1): extract a move once, then fit "
                   "it to any number of characters.")
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder",)
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"track": (TRACK_TYPE,), "folder": ("STRING", {"default": "pss_tracks"}),
                             "name": ("STRING", {"default": "track"})}}

    def save(self, track, folder, name):
        d = save_dir(folder, name)
        with open(os.path.join(d, "keypoints.json"), "w") as f:
            json.dump(track["keypoints"], f)
        if "feet" in track:
            with open(os.path.join(d, "feet.json"), "w") as f:
                json.dump(track["feet"], f)
        for key in ("contact", "cycle", "timing", "fit"):
            if key in track:
                with open(os.path.join(d, key + ".json"), "w") as f:
                    json.dump(track[key], f, indent=1)
        rel = os.path.relpath(d, output_dir())
        return {"ui": {"text": [f"saved: {rel}  (Load Skeleton Track folder: {rel})"]}, "result": (rel,)}


NODE_CLASS_MAPPINGS = {
    "PSS_ScanVideoPoses": PSS_ScanVideoPoses,
    "PSS_LoadPoseScan": PSS_LoadPoseScan,
    "PSS_SavePoseScan": PSS_SavePoseScan,
    "PSS_RepairPoseScan": PSS_RepairPoseScan,
    "PSS_TrackFromPoseScan": PSS_TrackFromPoseScan,
    "PSS_SaveSkeletonTrack": PSS_SaveSkeletonTrack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PSS_ScanVideoPoses": "Scan Video Poses (DWPose + feet)",
    "PSS_LoadPoseScan": "Load Pose Scan",
    "PSS_SavePoseScan": "Save Pose Scan",
    "PSS_RepairPoseScan": "Repair Pose Scan",
    "PSS_TrackFromPoseScan": "Track from Pose Scan",
    "PSS_SaveSkeletonTrack": "Save Skeleton Track",
}
