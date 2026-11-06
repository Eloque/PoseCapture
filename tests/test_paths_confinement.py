"""Path confinement checks for every file read and write of PoseCapture (1.2.3).

Builds a scratch ComfyUI layout (input/, output/, a sibling output2/ and an outside/ folder with a secret file), plants
symlinks that point out of input/ and output/, and attacks resolve() (all Load nodes, Scan Video Poses' video input,
Fit's keypoints json) and save_dir() (Save Pose Scan, Save Skeleton Track) through the real node classes too.

usage: python tests/test_paths_confinement.py      exit code 0 only if every check passes
"""
import importlib.util, json, os, sys, tempfile, types

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = tempfile.mkdtemp(prefix="pc_paths_")
INP, OUT, OUT2, OUTSIDE = (os.path.join(ROOT, d) for d in ("input", "output", "output2", "outside"))
for d in (INP, OUT, OUT2, OUTSIDE):
    os.makedirs(d)
SECRET = os.path.join(OUTSIDE, "secret.json")
json.dump({"secret": True}, open(SECRET, "w"))
os.makedirs(os.path.join(INP, "tracks", "ok"))
json.dump([], open(os.path.join(INP, "tracks", "ok", "keypoints.json"), "w"))
json.dump({"ok": True}, open(os.path.join(INP, "kp.json"), "w"))
json.dump({"sibling": True}, open(os.path.join(OUT2, "x.json"), "w"))
os.symlink(OUTSIDE, os.path.join(INP, "link_dir"))                 # a folder link out of input/
os.symlink(SECRET, os.path.join(INP, "link_file.json"))            # a file link out of input/
os.symlink(OUTSIDE, os.path.join(OUT, "link_out"))                 # a folder link out of output/ (write side)

fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: INP
fp.get_output_directory = lambda: OUT
fp.get_temp_directory = lambda: OUT
sys.modules["folder_paths"] = fp
spec = importlib.util.spec_from_file_location("posecapture", os.path.join(PACK, "__init__.py"),
                                              submodule_search_locations=[PACK])
mod = importlib.util.module_from_spec(spec)
sys.modules["posecapture"] = mod
spec.loader.exec_module(mod)
N, V = sys.modules["posecapture.nodes"], sys.modules["posecapture.nodes_video"]

results = []


def expect(desc, fn, ok):
    try:
        r = fn()
        passed = ok and not (isinstance(r, str) and not os.path.realpath(r).startswith((os.path.realpath(INP), os.path.realpath(OUT))))
        got = f"-> {os.path.relpath(r, ROOT) if isinstance(r, str) else 'ok'}"
    except ValueError as e:
        passed, got = (not ok), f"refused ({e})"
    except Exception as e:     # any other error is a failure: refusals must be clean ValueErrors
        passed, got = False, f"{type(e).__name__}: {e}"
    results.append(passed)
    print(("PASS " if passed else "FAIL ") + f"{desc}: {got}")


# ---- reads: resolve()
for p in ("tracks/ok/keypoints.json", "kp.json", "tracks/../kp.json", "tracks/ok"):
    expect(f"read allowed {p!r}", lambda p=p: N.resolve(p), True)
for p in (SECRET, "/etc/passwd", "../outside/secret.json", "tracks/../../outside/secret.json", "../output2/x.json",
          "link_dir/secret.json", "link_file.json", "link_dir", "\\\\host\\share\\x", "//host/share/x",
          "\\\\?\\C:\\x", "\\\\.\\pipe\\x", "kp.json\0.png", "", "   "):
    expect(f"read refused {p!r}", lambda p=p: N.resolve(p), False)

# ---- reads through the node classes
expect("Load Skeleton Track via a symlinked folder", lambda: N.PSS_LoadSkeletonTrack().load("link_dir"), False)
expect("Load Pose Scan via a symlinked file", lambda: V.PSS_LoadPoseScan().load("link_file.json", ""), False)
expect("Load Pose Scan, absolute path", lambda: V.PSS_LoadPoseScan().load(SECRET, ""), False)

# ---- writes: save_dir()
expect("write allowed pss_tracks/kick", lambda: V.save_dir("pss_tracks", "kick"), True)
expect("write allowed a/b/kick", lambda: V.save_dir("a/b", "kick"), True)
for f, n in (("../outside", "x"), (OUTSIDE, "x"), ("link_out", "x"), ("link_out/deeper", "x"), ("../output2", "x"),
             ("pss", "../x"), ("pss", "a/b"), ("pss", ".."), ("pss", "a\\b"), ("pss", ""), ("//host/share", "x")):
    expect(f"write refused folder={f!r} name={n!r}", lambda f=f, n=n: V.save_dir(f, n), False)
expect("nothing was written outside output/", lambda: (os.listdir(OUTSIDE) == ["secret.json"]) or (_ for _ in ()).throw(
    RuntimeError(f"outside/ now holds {os.listdir(OUTSIDE)}")), True)

print(f"\n{sum(results)}/{len(results)} passed  (scratch: {ROOT})")
sys.exit(0 if all(results) else 1)
