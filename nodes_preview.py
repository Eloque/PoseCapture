"""ComfyUI node: Preview Skeleton Track - plays any skeleton track (straight from Track from Pose Scan, loaded with Load
Skeleton Track, or fitted) as an animation inside the node, and returns the frames as IMAGE.

Colours as the other check pictures: R = near = orange, L = far = blue, head/torso white, green dot = planted foot,
grey line = ground. numpy + Pillow, no models.
"""
import os
import uuid

import numpy as np
from PIL import Image, ImageDraw

from .nodes import CATEGORY, TRACK_TYPE, frames_to_image, track_info
from .pss import preview as pss_prev

BACKGROUNDS = {"black": (0, 0, 0), "grey": (90, 90, 90), "white": (255, 255, 255)}
R75 = 0.75          # a fitted track's feet.json / contact.json are in render px = canvas x 0.75 (see pss/fit.py)


def temp_dir():
    try:
        import folder_paths
        return folder_paths.get_temp_directory()
    except Exception:
        return os.path.join(os.getcwd(), "temp")


def track_frames(track, height, background, show_feet, show_numbers, fit_figure=True):
    """list of HxWx3 uint8 frames of the track drawn at `height` px (0 = the track's canvas size); fit_figure: crop
    to the figure (one box over all frames) instead of the whole canvas"""
    kp = track["keypoints"]
    W, H = int(kp[0].get("canvas_width") or 1024), int(kp[0].get("canvas_height") or 1024)
    feet = (track.get("feet") or {}).get("frames") if show_feet else None
    fnote = (track.get("feet") or {}).get("note", "")
    fs = 1.0 / R75 if "render px" in fnote else 1.0        # feet.json scale -> canvas px
    ox = oy = 0.0
    if fit_figure:       # one box around the figure over all frames (so it does not jump), 8% margin
        P = [(x, y) for fr in kp for pp in (fr.get("people") or [])[:1]
             for x, y, sc in np.array(pp["pose_keypoints_2d"], float).reshape(-1, 3)[:18] if sc > 0.3 and x > 1]
        if feet:
            P += [(v[0] * fs, v[1] * fs) for f in feet for sd in ("right", "left") for v in f[sd].values()]
        if P:
            P = np.array(P)
            m = 0.08 * (P[:, 1].max() - P[:, 1].min())
            x0, y0 = P.min(0) - m
            x1, y1 = P.max(0) + m
            ox, oy, W, H = x0, y0, x1 - x0, y1 - y0
    k = height / H if height > 0 else 1.0
    w, h = max(2, round(W * k)), max(2, round(H * k))
    contact = (track.get("contact") or {}).get("frames")
    cnote = (track.get("contact") or {}).get("note", "")
    near = "left" if "near foot = left" in cnote else "right"
    sides = (near, "left" if near == "right" else "right")   # contact "planted" = [near, far]

    def foot(t, sd):
        f = feet[t][sd]
        return [((f[q][0] * fs - ox) * k, (f[q][1] * fs - oy) * k) for q in ("heel", "toe", "small")]

    def touch(t, sd):    # where the foot meets the floor: between heel and toe, at the lower of the two
        heel, toe, _ = foot(t, sd)
        return ((heel[0] + toe[0]) / 2, max(heel[1], toe[1]))

    ground = None
    if feet and contact and len(feet) == len(kp) and len(contact) == len(kp):
        ys = [touch(t, sides[i])[1] for t in range(len(kp)) for i in (0, 1) if contact[t]["planted"][i]]
        ground = float(np.median(ys)) if ys else None

    bg = np.zeros((h, w, 3), np.uint8)
    bg[:] = BACKGROUNDS[background]
    out = []
    for t, fr in enumerate(kp):
        people = fr.get("people") or []
        if not people:
            out.append(bg.copy())
            continue
        a = np.array(people[0]["pose_keypoints_2d"], float).reshape(-1, 3)[:18]
        body = [[(x - ox) * k, (y - oy) * k, s] for x, y, s in a]
        fp = planted = pts = None
        if feet and t < len(feet):
            fp = {sd: foot(t, sd) for sd in ("right", "left")}
            if contact and t < len(contact):
                planted = contact[t]["planted"]
                pts = [touch(t, sides[0]), touch(t, sides[1])]
        img = pss_prev.draw_pose(bg, body, fp, planted, pts, ground, alpha_bg=1.0)
        if show_numbers:
            im = Image.fromarray(img)
            col = (0, 0, 0) if background == "white" else (220, 220, 220)
            ImageDraw.Draw(im).text((6, 4), f"{t}", fill=col)
            img = np.asarray(im)
        out.append(np.ascontiguousarray(img))
    return out


class PSS_PreviewSkeletonTrack:
    DESCRIPTION = ("Plays a skeleton track as an animation inside the node: connect any track (from Track from Pose "
                   "Scan, Load Skeleton Track or Fit Track to Character). R = near = orange, L = far = blue, green dot = "
                   "planted foot, grey line = ground, frame number top left. Also returns the frames as IMAGE (e.g. for "
                   "Video Combine).")
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "info")
    FUNCTION = "preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "track": (TRACK_TYPE,),
            "fps": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 120.0, "step": 1.0,
                              "tooltip": "playback speed of the preview"}),
            "height": ("INT", {"default": 640, "min": 0, "max": 4096, "step": 8,
                               "tooltip": "preview height in px; 0 = full size"}),
            "framing": (["fit figure", "whole canvas"], {"default": "fit figure",
                         "tooltip": "fit figure: one crop around the figure over all frames; whole canvas: the track's canvas"}),
            "background": (list(BACKGROUNDS), {"default": "black"}),
            "show_feet": ("BOOLEAN", {"default": True, "tooltip": "feet triangles, planted dots and the ground line"}),
            "frame_numbers": ("BOOLEAN", {"default": True}),
        }}

    def preview(self, track, fps, height, framing, background, show_feet, frame_numbers):
        frames = track_frames(track, height, background, show_feet, frame_numbers, framing == "fit figure")
        d = temp_dir()
        os.makedirs(d, exist_ok=True)
        name = f"pss_track_preview_{uuid.uuid4().hex[:10]}.webp"
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(os.path.join(d, name), save_all=True, append_images=ims[1:], duration=round(1000 / fps), loop=0,
                    quality=85, method=4)
        ui = {"images": [{"filename": name, "subfolder": "", "type": "temp"}], "animated": (len(ims) > 1,)}
        return {"ui": ui, "result": (frames_to_image(frames), track_info(track))}


NODE_CLASS_MAPPINGS = {"PSS_PreviewSkeletonTrack": PSS_PreviewSkeletonTrack}
NODE_DISPLAY_NAME_MAPPINGS = {"PSS_PreviewSkeletonTrack": "Preview Skeleton Track"}
