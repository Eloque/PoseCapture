"""Gate for the video nodes through a running ComfyUI (API graph): the walk reference video -> Scan -> Track (walk
cycle) -> Fit -> Render, and the stored scan of the same video -> Track -> Fit -> Render, compared with the workflow
bundle's walk track and walk pose frames.
  python test_video_graph.py SERVER VIDEO STORED_SCAN BUNDLE_DIR TAG REPORT.json COMFY_OUTPUT_DIR
"""
import json, os, sys, time, urllib.request
import numpy as np
from PIL import Image

SERVER, VIDEO, SCAN, BUNDLE, TAG, REPORT, OUTDIR = sys.argv[1:8]


def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


KP = os.path.join(BUNDLE, "inputs/scail_cop/cop/cop_keypoints.json")
TRACK = dict(mode="walk cycle", start=49, frames=57, period=26, face="auto", near_side="auto", remove_travel="auto",
             far_drop=-1.0, ground_tol=12.0, swing_tol=4.0, smooth=0.0, preview_height=720)
FIT = dict(char_keypoints_json=KP, canvas=1024, hip_x=347.0, fixed_shift=True, pad_top=0, scale_frame=0)
REN = dict(size=768, crop_h=0, crop_off=-1, height_mm=1550.0, shoulder_mm=200.0, hip_mm=120.0, min_sep_mm=60.0,
           radius_mm=21.5, fov=55.0)
VC = dict(frame_rate=25, loop_count=0, format="video/h264-mp4", pingpong=False, save_output=True)
g = {
    # branch A: from the video
    "1": {"class_type": "PSS_ScanVideoPoses", "inputs": dict(video=VIDEO, crop="0,0,640,720", start=0, frames=0, every=1,
          bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt", overlay_height=720)},
    "2": {"class_type": "PSS_SavePoseScan", "inputs": dict(scan=["1", 0], folder=f"{TAG}/scan", name="walk")},
    "3": {"class_type": "PSS_TrackFromPoseScan", "inputs": dict(scan=["1", 0], **TRACK)},
    "4": {"class_type": "PSS_SaveSkeletonTrack", "inputs": dict(track=["3", 0], folder=f"{TAG}/track", name="walk")},
    "5": {"class_type": "PSS_FitTrackToCharacter", "inputs": dict(track=["3", 0], **FIT)},
    "6": {"class_type": "PSS_RenderSCAILCylinders", "inputs": dict(track=["5", 0], **REN)},
    "7": {"class_type": "SaveImage", "inputs": dict(images=["6", 0], filename_prefix=f"{TAG}/A_pose/pose")},
    "8": {"class_type": "VHS_VideoCombine", "inputs": dict(images=["3", 1], filename_prefix=f"{TAG}/A_track_preview", **VC)},
    "9": {"class_type": "VHS_VideoCombine", "inputs": dict(images=["1", 1], filename_prefix=f"{TAG}/A_scan_overlay", **VC)},
    "10": {"class_type": "VHS_VideoCombine", "inputs": dict(images=["6", 0], filename_prefix=f"{TAG}/A_pose_video", **VC)},
    # branch B: from the stored scan
    "11": {"class_type": "PSS_LoadPoseScan", "inputs": dict(path=SCAN, video=VIDEO)},
    "12": {"class_type": "PSS_TrackFromPoseScan", "inputs": dict(scan=["11", 0], **TRACK)},
    "13": {"class_type": "PSS_SaveSkeletonTrack", "inputs": dict(track=["12", 0], folder=f"{TAG}/track", name="walk_stored")},
    "14": {"class_type": "PSS_FitTrackToCharacter", "inputs": dict(track=["12", 0], **FIT)},
    "15": {"class_type": "PSS_RenderSCAILCylinders", "inputs": dict(track=["14", 0], **REN)},
    "16": {"class_type": "SaveImage", "inputs": dict(images=["15", 0], filename_prefix=f"{TAG}/B_pose/pose")},
}
t0 = time.time()
pid = post(f"{SERVER}/prompt", {"prompt": g})["prompt_id"]
while True:
    h = get(f"{SERVER}/history/{pid}")
    if pid in h and h[pid].get("status", {}).get("completed") is not None:
        break
    time.sleep(5)
st = h[pid]["status"]
if st.get("status_str") != "success":
    print(json.dumps(st, indent=1)[:4000])
    sys.exit(1)
out = h[pid]["outputs"]


def frames(node, root):
    return [os.path.join(root, im["subfolder"], im["filename"]) for im in out[node]["images"]]


rep = {"seconds": round(time.time() - t0), "prompt_id": pid}
ref_frames = sorted(os.listdir(os.path.join(BUNDLE, "inputs/scail_cop/poses/walk")))
ref = [np.asarray(Image.open(os.path.join(BUNDLE, "inputs/scail_cop/poses/walk", f)).convert("RGB")) for f in ref_frames]
src_track = os.path.join(BUNDLE, "inputs/tracks/walk/source")
for br, node, tnode in (("A_video", "7", "4"), ("B_stored_scan", "16", "13")):
    ours = [np.asarray(Image.open(p).convert("RGB")) for p in frames(node, OUTDIR)]
    diffs = [int((np.abs(a.astype(int) - b.astype(int)) > 0).any(-1).sum()) for a, b in zip(ours, ref)]
    tdir = out[tnode]["text"][0]
    kp_same = open(os.path.join(tdir, "keypoints.json")).read() == open(os.path.join(src_track, "keypoints.json")).read()
    feet_same = json.load(open(os.path.join(tdir, "feet.json")))["frames"] == json.load(open(os.path.join(src_track, "feet.json")))["frames"]
    con_same = json.load(open(os.path.join(tdir, "contact.json")))["frames"] == json.load(open(os.path.join(src_track, "contact.json")))["frames"]
    a = np.array([f["people"][0]["pose_keypoints_2d"] for f in json.load(open(os.path.join(tdir, "keypoints.json")))]).reshape(57, 18, 3)
    b = np.array([f["people"][0]["pose_keypoints_2d"] for f in json.load(open(os.path.join(src_track, "keypoints.json")))]).reshape(57, 18, 3)
    rep[br] = {"frames": len(ours), "identical_frames": sum(d == 0 for d in diffs), "max_px_differing": max(diffs),
               "mean_px_differing": round(float(np.mean(diffs)), 1), "track_keypoints_bytes_identical": kp_same,
               "track_feet_equal": feet_same, "track_contact_equal": con_same,
               "track_max_point_diff_px": round(float(np.abs(a[:, :, :2] - b[:, :, :2]).max()), 3),
               "track_mean_point_diff_px": round(float(np.abs(a[:, :, :2] - b[:, :, :2]).mean()), 3), "track_dir": tdir}
rep["outputs"] = {k: v for k, v in out.items()}
json.dump(rep, open(REPORT, "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "outputs"}, indent=1))
