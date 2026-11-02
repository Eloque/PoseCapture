"""ComfyUI nodes: skeleton track -> fitted track -> SCAIL cylinder pose frames, and a still padded onto the pose box.

CPU only, numpy + Pillow, no models. The computing code lives in pss/ (copied unchanged from the workflow's scripts);
this file only converts between ComfyUI types and the scripts' data.
"""
import json
import os

import numpy as np
import torch
from PIL import Image

from .pss import fit as pss_fit
from .pss import cylinders as pss_cyl
from .pss import pad as pss_pad

CATEGORY = "Posed Sprite Sheet"
TRACK_TYPE = "PSS_SCAIL_TRACK"
TRACK_FILES = ("keypoints", "feet", "contact", "timing", "cycle", "fit")


# ---------------------------------------------------------------- helpers
def input_dir():
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:          # outside ComfyUI (tests): relative paths are relative to the working directory
        return os.getcwd()


def output_dir_or_none():
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        return None


def resolve(path):
    """absolute path as given; a relative path is looked up in the ComfyUI input folder, then in the output folder
    (so a track saved by Save Skeleton Track loads with the folder name it printed, e.g. pss_tracks/kick_v1)"""
    path = path.strip().strip('"')
    if os.path.isabs(path):
        return path
    p = os.path.join(input_dir(), path)
    out = output_dir_or_none()
    if not os.path.exists(p) and out and os.path.exists(os.path.join(out, path)):
        return os.path.join(out, path)
    return p


def frames_to_image(frames):
    """list of HxWx3 uint8 arrays -> IMAGE (B,H,W,3) float32 0..1 (exact: x/255*255 gives x back for every uint8)"""
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def image_to_uint8(image):
    """IMAGE (B,H,W,C) float 0..1 -> (B,H,W,C) uint8, the inverse of frames_to_image"""
    return np.clip(np.round(255.0 * image.detach().cpu().numpy()), 0, 255).astype(np.uint8)


def image_to_pil(image, i=0):
    a = image_to_uint8(image[i:i + 1])[0]
    if a.shape[2] == 1:
        a = a[:, :, 0]
    return Image.fromarray(a)


def load_track_folder(folder):
    d = resolve(folder)
    if not os.path.isfile(os.path.join(d, "keypoints.json")):
        raise FileNotFoundError(f"no keypoints.json in the track folder '{folder}'")
    track = {"name": os.path.basename(os.path.normpath(d))}
    for key in TRACK_FILES:
        p = os.path.join(d, key + ".json")
        if os.path.isfile(p):
            with open(p) as f:
                track[key] = json.load(f)
    return track


def track_info(track):
    kp = track["keypoints"]
    files = ", ".join(k + ".json" for k in TRACK_FILES if k in track)
    return (f"track '{track['name']}': {len(kp)} frames, canvas {kp[0].get('canvas_width')}x{kp[0].get('canvas_height')}"
            f"; files: {files}")


# ---------------------------------------------------------------- nodes
class PSS_LoadSkeletonTrack:
    DESCRIPTION = ("Loads a skeleton track folder: keypoints.json (OpenPose-18 per frame, DWPose json format) plus, if "
                   "present, feet.json, contact.json, timing.json, cycle.json, fit.json. Works for a source track (input "
                   "of Fit Track to Character) and a fitted one (input of Render SCAIL Cylinders). A relative folder is "
                   "read from the ComfyUI input folder.")
    CATEGORY = CATEGORY
    RETURN_TYPES = (TRACK_TYPE, "INT", "STRING")
    RETURN_NAMES = ("track", "frames", "info")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"folder": ("STRING", {"default": "tracks/short_uppercut/fitted",
                                                   "tooltip": "track folder: relative to ComfyUI/input, else ComfyUI/output (where Save Skeleton Track writes), or absolute"})}}

    @classmethod
    def IS_CHANGED(cls, folder):
        d = resolve(folder)
        stamp = []
        for key in TRACK_FILES:
            p = os.path.join(d, key + ".json")
            if os.path.isfile(p):
                st = os.stat(p)
                stamp.append(f"{key}:{st.st_mtime_ns}:{st.st_size}")
        return "|".join(stamp)

    def load(self, folder):
        track = load_track_folder(folder)
        return (track, len(track["keypoints"]), track_info(track))


class PSS_FitTrackToCharacter:
    DESCRIPTION = ("Re-proportions a source skeleton track to a character: every bone keeps its direction per frame and "
                   "takes the character's bone length (measured on the character's DWPose keypoints), the move is "
                   "placed on a canvas of width 'canvas' and height 1024 + pad_top, and the hips are held at hip_x "
                   "(fixed_shift: one shift for the whole clip, for walks). Character keypoints: connect a DWPose "
                   "POSE_KEYPOINT, or give the keypoints json file (relative to ComfyUI/input).")
    CATEGORY = CATEGORY
    RETURN_TYPES = (TRACK_TYPE, "IMAGE", "STRING")
    RETURN_NAMES = ("fitted_track", "check", "fit_info")
    FUNCTION = "fit"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "track": (TRACK_TYPE, {"tooltip": "source track (needs keypoints, feet and contact)"}),
                    "char_keypoints_json": ("STRING", {"default": "scail_cop/cop/cop_keypoints.json",
                                                       "tooltip": "used when char_keypoints is not connected"}),
                    "canvas": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 1,
                                       "tooltip": "canvas width; the height is 1024 + pad_top"}),
                    "hip_x": ("FLOAT", {"default": 384.0, "min": -8192.0, "max": 8192.0, "step": 1.0, "round": False}),
                    "fixed_shift": ("BOOLEAN", {"default": False,
                                                "tooltip": "one shift for every frame (walks keep the feet locked)"}),
                    "pad_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                                        "tooltip": "extra canvas above the 1024 px (moves that rise high)"}),
                    "scale_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                            "tooltip": "a STANDING frame for clips that start crouched"}),
                },
                "optional": {"char_keypoints": ("POSE_KEYPOINT", {"tooltip": "DWPose output of the character picture"})}}

    def fit(self, track, char_keypoints_json, canvas, hip_x, fixed_shift, pad_top, scale_frame, char_keypoints=None):
        for key in ("feet", "contact"):
            if key not in track:
                raise ValueError(f"track '{track['name']}' has no {key}.json: Fit needs a source track folder")
        if char_keypoints is None:
            with open(resolve(char_keypoints_json)) as f:
                char_keypoints = json.load(f)
        if isinstance(char_keypoints, dict):
            char_keypoints = [char_keypoints]
        out, check, warnings, summary = pss_fit.fit_track(track, char_keypoints, canvas=canvas, hip_x=hip_x,
                                                          fixed_shift=fixed_shift, pad_top=pad_top,
                                                          scale_frame=scale_frame)
        out["name"] = track["name"] + " (fitted)"
        text = "\n".join(warnings + [summary, json.dumps(out["fit"], indent=1)])
        return (out, frames_to_image([np.asarray(check)]), text)


class PSS_RenderSCAILCylinders:
    DESCRIPTION = ("Renders the SCAIL-Preview pose frames from a fitted track: z-buffered 3D cylinders (13 bones, "
                   "SCAIL-Pose colours and shading, black background). OpenPose R side = near the camera. "
                   "size = output width; crop_h > 0 = output height (a window of the square canvas), crop_off = its "
                   "top (-1 = centred). CPU, about 1-2 s per frame. params = the camera numbers as JSON "
                   "(camera.mm_per_px gives the sprite-sheet scale between moves).")
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_frames", "params")
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo, hi, st: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": st, "round": False})
        return {"required": {
            "track": (TRACK_TYPE, {"tooltip": "fitted track"}),
            "size": ("INT", {"default": 768, "min": 16, "max": 8192, "step": 1}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, "tooltip": "0 = full height"}),
            "crop_off": ("INT", {"default": -1, "min": -1, "max": 8192, "step": 1, "tooltip": "-1 = centred"}),
            "height_mm": f(1550.0, 1.0, 10000.0, 1.0),
            "shoulder_mm": f(200.0, 0.0, 2000.0, 0.5),
            "hip_mm": f(120.0, 0.0, 2000.0, 0.5),
            "min_sep_mm": f(60.0, 0.0, 2000.0, 0.5),
            "radius_mm": f(21.5, 0.5, 500.0, 0.5),
            "fov": f(55.0, 1.0, 170.0, 0.5),
        }}

    def render(self, track, size, crop_h, crop_off, height_mm, shoulder_mm, hip_mm, min_sep_mm, radius_mm, fov):
        progress = None
        try:
            import comfy.utils
            bar = comfy.utils.ProgressBar(len(track["keypoints"]))
            progress = lambda i, n: bar.update_absolute(i, n)
        except Exception:
            pass
        frames, params, _ = pss_cyl.render_track(track["keypoints"], track_name=track["name"], size=size,
                                                 crop_h=crop_h, crop_off=None if crop_off < 0 else crop_off,
                                                 height_mm=height_mm, shoulder_mm=shoulder_mm, hip_mm=hip_mm,
                                                 min_sep_mm=min_sep_mm, radius_mm=radius_mm, fov=fov,
                                                 progress=progress)
        return (frames_to_image(frames), json.dumps(params, indent=1))


class PSS_PadStillToPose:
    DESCRIPTION = ("Crops the figure out of a side still on plain white, scales it to the pose figure's height and "
                   "pastes it on a white 768x768 canvas, so SCAIL starts from the same framing as pose frame 0. "
                   "cx = horizontal centre, y0/y1 = top/bottom of the figure on the 768 canvas. keep_size: no "
                   "resampling, a bigger square canvas instead. mode 'auto' reads cx/y0/y1 off pose frame 0 "
                   "(connect pose_frames) with the head/feet margins.")
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("padded", "info")
    FUNCTION = "pad"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "image": ("IMAGE", {"tooltip": "side still, figure on plain white"}),
                    "cx": ("FLOAT", {"default": 384.0, "min": -4096.0, "max": 4096.0, "step": 0.5, "round": False}),
                    "y0": ("INT", {"default": 40, "min": -4096, "max": 4096, "step": 1}),
                    "y1": ("INT", {"default": 691, "min": -4096, "max": 4096, "step": 1}),
                    "keep_size": ("BOOLEAN", {"default": False}),
                    "mode": (["manual", "auto from pose frame 0"], {"default": "manual"}),
                },
                "optional": {
                    "pose_frames": ("IMAGE", {"tooltip": "only for mode auto: the pose frames (frame 0 is used)"}),
                    "head_margin": ("FLOAT", {"default": 0.067, "min": 0.0, "max": 1.0, "step": 0.001, "round": False}),
                    "feet_margin": ("FLOAT", {"default": 0.022, "min": 0.0, "max": 1.0, "step": 0.001, "round": False}),
                }}

    def pad(self, image, cx, y0, y1, keep_size, mode, pose_frames=None, head_margin=0.067, feet_margin=0.022):
        lines = []
        if mode != "manual":
            if pose_frames is None:
                raise ValueError("mode auto needs pose_frames connected")
            cx, y0, y1, box = pss_pad.box_from_pose_frame(image_to_uint8(pose_frames[0:1])[0], head_margin, feet_margin)
            lines.append(f"auto: pose frame-0 box x {box[0]}-{box[2]} y {box[1]}-{box[3]} -> cx {cx} y0 {y0} y1 {y1}")
        outs = []
        for i in range(image.shape[0]):
            im, info = pss_pad.pad_still(image_to_pil(image, i), cx, y0, y1, keep_size)
            outs.append(np.asarray(im))
            lines.append(info)
        if len({o.shape for o in outs}) > 1:
            raise ValueError("the stills of the batch give different padded sizes: pad them one at a time")
        return (frames_to_image(outs), "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "PSS_LoadSkeletonTrack": PSS_LoadSkeletonTrack,
    "PSS_FitTrackToCharacter": PSS_FitTrackToCharacter,
    "PSS_RenderSCAILCylinders": PSS_RenderSCAILCylinders,
    "PSS_PadStillToPose": PSS_PadStillToPose,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PSS_LoadSkeletonTrack": "Load Skeleton Track",
    "PSS_FitTrackToCharacter": "Fit Track to Character",
    "PSS_RenderSCAILCylinders": "Render SCAIL Cylinders",
    "PSS_PadStillToPose": "Pad Still to Pose",
}
