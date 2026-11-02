"""Exactness gate through a real ComfyUI server: queue one API-format graph with the full chain for all three moves
(Load Skeleton Track (source) -> Fit Track to Character -> Render SCAIL Cylinders -> Save Image; Load Image -> Pad Still
to Pose -> Save Image), wait, and compare the saved PNGs with the bundle's files.

usage: python tests/test_comfy_graph.py BUNDLE_DIR SERVER_OUTPUT_DIR RUN_TAG [--url http://127.0.0.1:8191] [--out report.json]
  The server's input folder must hold the bundle's inputs/tracks and inputs/scail_cop (as tracks/, scail_cop/).
  RUN_TAG = a new output sub-folder name (results land in SERVER_OUTPUT_DIR/RUN_TAG/<move>/).
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_exact import MOVES, RENDER_DEFAULTS  # noqa: E402


def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def build(tag):
    g = {}
    for k, (move, cfg) in enumerate(MOVES.items()):
        b = 100 * (k + 1)
        g[str(b + 1)] = {"class_type": "PSS_LoadSkeletonTrack", "inputs": {"folder": f"tracks/{move}/source"}}
        g[str(b + 2)] = {"class_type": "PSS_FitTrackToCharacter",
                         "inputs": {"track": [str(b + 1), 0], "char_keypoints_json": "scail_cop/cop/cop_keypoints.json",
                                    **cfg["fit"]}}
        g[str(b + 3)] = {"class_type": "PSS_RenderSCAILCylinders",
                         "inputs": {"track": [str(b + 2), 0], **RENDER_DEFAULTS, **cfg["render"]}}
        g[str(b + 4)] = {"class_type": "SaveImage", "inputs": {"images": [str(b + 3), 0], "filename_prefix": f"{tag}/{move}/pose"}}
        p = cfg["pad"]
        g[str(b + 5)] = {"class_type": "LoadImage", "inputs": {"image": f"scail_cop/refs/{p['still']}"}}
        g[str(b + 6)] = {"class_type": "PSS_PadStillToPose",
                         "inputs": {"image": [str(b + 5), 0], "cx": p["cx"], "y0": p["y0"], "y1": p["y1"],
                                    "keep_size": p["keep_size"], "mode": "manual", "head_margin": 0.067, "feet_margin": 0.022}}
        g[str(b + 7)] = {"class_type": "SaveImage", "inputs": {"images": [str(b + 6), 0], "filename_prefix": f"{tag}/{move}/padded"}}
    return g


def cmp(ours, refs):
    same, maxd = 0, 0
    for a, b in zip(ours, refs):
        A = np.asarray(Image.open(a).convert("RGB")).astype(int)
        B = np.asarray(Image.open(b).convert("RGB")).astype(int)
        if A.shape != B.shape:
            maxd = max(maxd, 999)
            continue
        d = int(np.abs(A - B).max())
        maxd = max(maxd, d)
        same += d == 0
    return {"identical": same, "total": len(refs), "count_match": len(ours) == len(refs), "max_diff": maxd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("server_output")
    ap.add_argument("tag")
    ap.add_argument("--url", default="http://127.0.0.1:8191")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.server_output, a.tag)):
        sys.exit("tag exists: new tag per run")
    t0 = time.time()
    pid = post(f"{a.url}/prompt", {"prompt": build(a.tag)})["prompt_id"]
    while True:
        time.sleep(5)
        h = get(f"{a.url}/history/{pid}")
        if pid in h and (h[pid].get("status", {}).get("completed") or h[pid].get("status", {}).get("status_str") == "error"):
            status = h[pid]["status"]
            break
    rep = {"prompt_id": pid, "status": status.get("status_str"), "seconds": round(time.time() - t0, 1), "moves": {}}
    B = os.path.join(a.bundle, "inputs", "scail_cop")
    for move, cfg in MOVES.items():
        D = os.path.join(a.server_output, a.tag, move)
        ours = sorted(os.path.join(D, f) for f in os.listdir(D) if f.startswith("pose_"))
        refs = sorted(os.path.join(B, "poses", move, f) for f in os.listdir(os.path.join(B, "poses", move)))
        pads = sorted(os.path.join(D, f) for f in os.listdir(D) if f.startswith("padded_"))
        rep["moves"][move] = {"pose": cmp(ours, refs), "pad": cmp(pads, [os.path.join(B, "refs", cfg["pad"]["ref"])])}
    rep["all_exact"] = all(r[k]["identical"] == r[k]["total"] and r[k]["count_match"] for r in rep["moves"].values() for k in r)
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    sys.exit(0 if rep["all_exact"] else 1)


if __name__ == "__main__":
    main()
