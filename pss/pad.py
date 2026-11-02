"""Pad a character still (figure on plain white) onto the pose video's frame-0 box.

Logic copied unchanged from the workflow bundle's scripts/prep_ref_pad.py; only the file I/O moved out.
SCAIL keeps the reference image's own framing: if the figure in the still is not where the pose figure stands in
frame 0, the render drifts and the feet run off the bottom. So the still is cropped to the figure, scaled to the
height of the pose figure's box and pasted on a white 768x768 canvas at the right place.

Look at pose frame 0: the pose figure's box (x/y of the coloured cylinders) tells you the numbers. The ones used for
the example cop (pose frame-0 box -> still):
  short uppercut  pose box y 80-678   ->  cx 289 y0 40  y1 691     (head 40 px above the box, feet 13 below)
  walk            pose box y 68-666   ->  cx 249 y0 28  y1 679
  heavy uppercut  pose box y 233-726  ->  keep_size cx 284 y0 200 y1 737
keep_size: do not resample the figure; make the canvas bigger instead (square S x S, S >= 768) so that, scaled to
768, the figure lands on y0-y1. ComfyUI scales the image to the model size anyway. (Used for the heavy uppercut.)
"""
import numpy as np
from PIL import Image


def pad_still(im, cx, y0, y1, keep_size=False):
    """im: PIL image of the side still. Returns (padded PIL RGB image, info text)."""
    im = im.convert("RGB")
    v = np.asarray(im).astype(int)
    ys, xs = np.nonzero(v.min(2) < 225)                     # the figure = everything that is not near-white
    if len(ys) == 0:
        raise ValueError("no figure found: the still must show the figure on plain white (pixels < 225)")
    if keep_size:
        fy0, fy1, fx0, fx1 = ys.min(), ys.max(), xs.min(), xs.max()
        S = round(768 * (fy1 - fy0 + 1) / (y1 - y0 + 1))
        cxs = (fx0 + fx1) / 2
        ox = round(cx * S / 768 - cxs)
        oy = round(y0 * S / 768 - fy0)
        S = max(S, 768)
        c = Image.new("RGB", (S, S), (255, 255, 255))
        c.paste(im, (ox, oy))
        info = f"{S}x{S}, figure y {(fy0 + oy) * 768 / S:.0f}-{(fy1 + oy) * 768 / S:.0f}, x {(fx0 + ox) * 768 / S:.0f}-{(fx1 + ox) * 768 / S:.0f}"
    else:
        im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        H = y1 - y0 + 1
        W = round(im.width * H / im.height)
        im = im.resize((W, H), Image.LANCZOS)
        c = Image.new("RGB", (768, 768), (255, 255, 255))
        x0 = round(cx - W / 2)
        c.paste(im, (x0, y0))
        info = f"figure {W}x{H} at x {x0}-{x0 + W}, y {y0}-{y1}"
    return c, info


def box_from_pose_frame(frame, head_margin=0.067, feet_margin=0.022, threshold=0):
    """Optional helper (the node's "auto" mode, off by default): read cx, y0, y1 off a pose frame (HxWx3 uint8,
    black background). cx = centre of the cylinders' x range (rounded down), y0 = box top minus head_margin x box
    height, y1 = box bottom plus feet_margin x box height. The margins cover the head above the head cylinder and the
    boots below the ankle; the defaults are the ones that give the example cop's numbers on all three moves."""
    ys, xs = np.nonzero(np.asarray(frame).max(2) > threshold)
    if len(ys) == 0:
        raise ValueError("pose frame is empty (all black)")
    top, bot, left, right = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h = bot - top
    return (left + right) // 2, top - round(head_margin * h), bot + round(feet_margin * h), (left, top, right, bot)
