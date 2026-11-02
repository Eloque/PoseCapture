# PoseCapture: Posed Sprite Sheet Generator

**Turn video motion into game-ready character sprite sheets, right inside ComfyUI.**

Source: https://github.com/Eloque/PoseCapture

ComfyUI nodes that turn a video of a person into the pose video SCAIL-Preview needs, fitted to your own character:
z-buffered 3D cylinders, so the near arm and leg stay in front of the far ones on every frame of a side view.

    video -> Scan Video Poses -> Repair Pose Scan -> Track from Pose Scan -> Fit Track to Character -> Render SCAIL Cylinders

Only the scan uses a model (DWPose, from comfyui_controlnet_aux); every other node is CPU numpy + Pillow.

They are the Python steps of the Civitai workflow "Posed Sprite Sheet Generator - V2"
(https://civitai.com/models/3021972, video https://youtu.be/iD2Oxl6XVws) as nodes, so a character picture -> SCAIL
render runs as one graph. The computing code is the workflow's scripts (`fit_track.py`, `scail_pose_from_track.py`,
`prep_ref_pad.py`), copied with unchanged logic; on the workflow's example inputs the nodes give the same fitted tracks
(JSON equal), the same 187 pose frames (pixel-identical) and the same 3 padded stills (pixel-identical) as the scripts.

How this differs from other SCAIL pose packs: `ComfyUI-SCAIL-Pose` (and similar) estimate 3D poses from a driving
video with a model (NLF) and render those. This pack measures a 2D side-view skeleton track (from a video, or built
key pose by key pose), cleans it for a 2D game (one ground line, planted feet that do not slide, near side toward the
camera), re-proportions it to the character and renders the cylinders from it.

## Nodes

### From a video to a skeleton track

| Node (class) | In | Out |
|---|---|---|
| **Scan Video Poses (DWPose + feet)** (`PSS_ScanVideoPoses`) | `video` (file, relative to `ComfyUI/input/` or absolute; or connect `images`), `crop` (`x,y,w,h`, empty = whole frame), `start`, `frames` (0 = to the end), `every`, `bbox_detector`, `pose_estimator`, `overlay_height`. Runs DWPose whole-body per frame and keeps the 18 body points AND the 6 foot points (heel, big toe, small toe), which the usual DWPose keypoints drop. About 1 s per frame on a CPU. | `scan` (`PSS_POSE_SCAN`), `overlay` (IMAGE: the scan over the video), `info` |
| **Load Pose Scan** / **Save Pose Scan** (`PSS_LoadPoseScan`, `PSS_SavePoseScan`) | a scan saved as `keypoints_feet.json` (Save writes `ComfyUI/output/<folder>/<name>_vN/`, a new folder every time) | `scan` / the file path |
| **Repair Pose Scan** (`PSS_RepairPoseScan`) | `scan`; `fix_swaps` (DWPose swaps whole limbs for a few frames when the legs cross in a side view: a Viterbi pass picks the labelling with the smoothest motion), `interpolate` (lost points + a 3-frame median), `clamp_feet` (heel/toe points flung away from the ankle), `flip_cost` (px of smoothness a swapped frame must gain, -1 = 0.5% of the figure height: without it a stretch between two moments where the arms cross could be relabelled for free) | `scan`, `info` (which frames were swapped, how many points mended) |
| **Track from Pose Scan** (`PSS_TrackFromPoseScan`) | `scan`, `mode`: **walk cycle** (one cycle `start`..`start+period-1` closed into a seamless loop, planted feet moving back at one treadmill speed; a walk across the frame is put on a treadmill first; `period` 0 = found automatically) or **one-shot** (frames `start`..`start+frames-1` as they are, planted feet held still: punches, kicks, step-ins). `face` / `near_side` (auto), `remove_travel`, `far_drop`, `ground_tol`, `swing_tol`, `smooth`. | `track` (`PSS_SCAIL_TRACK`, a source track for Fit), `preview` (IMAGE: the cleaned track over the video; R = near = orange, L = far = blue, green dot = planted foot), `frames`, `info` |
| **Save Skeleton Track** (`PSS_SaveSkeletonTrack`) | `track` | a track folder (keypoints, feet, contact, cycle json) that Load Skeleton Track reads |
| **Preview Skeleton Track** (`PSS_PreviewSkeletonTrack`) | any `track` (from Track from Pose Scan, Load Skeleton Track or Fit), `fps`, `height`, `framing` (*fit figure*: one crop around the figure over all frames; *whole canvas*), `background`, `show_feet`, `frame_numbers` | plays the poses as an animation inside the node (R = near = orange, L = far = blue, green dot = planted foot, ground line, frame number); `frames` (IMAGE), `info` |

What Track from Pose Scan does to the measured motion: the figure is mirrored to face right if needed and its limbs
relabelled so the side toward the camera is R (the convention below); the far foot, which plants higher on screen
(perspective), is moved down onto the near foot's ground line; a heel or toe that is near the ground and not swinging
forward counts as down, and a foot is planted while its heel or toe is down; planted points stay exactly on the ground
line without sliding or jitter, and the correction fades in and out over two frames so the swing keeps the video's
motion. The walk-cycle mode is the cleanup that made the workflow's walk track: on the same scan it gives the same
track byte for byte.

**A good video:** one person, seen from the side, whole body in frame, static camera, plain background. A clear,
deliberate move: big motions (walk, punch, kick) come out well, tiny ones drown in DWPose's pixel jitter. For a
walk cycle the clip must hold at least two whole steps after `start` plus one frame.

### From a skeleton track to the SCAIL pose video

| Node (class) | In | Out |
|---|---|---|
| **Load Skeleton Track** (`PSS_LoadSkeletonTrack`) | `folder`: a track folder, relative to `ComfyUI/input/` or absolute. Reads `keypoints.json` (OpenPose-18 per frame, DWPose json format) and, if present, `feet.json`, `contact.json`, `timing.json`, `cycle.json`, `fit.json`. | `track` (`PSS_SCAIL_TRACK`), `frames` (INT), `info` |
| **Fit Track to Character** (`PSS_FitTrackToCharacter`) | `track` (a source track: needs keypoints, feet and contact), the character's keypoints: `char_keypoints` (POSE_KEYPOINT from comfyui_controlnet_aux's DWPose node) or `char_keypoints_json` (a json file, used when nothing is connected), `canvas` (width; height = 1024 + `pad_top`), `hip_x`, `fixed_shift` (one shift for the whole clip, for walks), `pad_top` (room above for moves that rise high), `scale_frame` (a standing frame, for clips that start crouched). | `fitted_track`, `check` (IMAGE: the character's skeleton + 6 fitted frames), `fit_info` (the numbers + warnings) |
| **Render SCAIL Cylinders** (`PSS_RenderSCAILCylinders`) | `track` (fitted), `size` (output width, 768), `crop_h` (0 = full height; else the output height, a window of the canvas), `crop_off` (top of that window, -1 = centred), `height_mm` 1550, `shoulder_mm` 200, `hip_mm` 120, `min_sep_mm` 60, `radius_mm` 21.5, `fov` 55. | `pose_frames` (IMAGE batch, RGB on black: the SCAIL pose video), `params` (JSON: camera, `mm_per_px`, depth range ...) |
| **Pad Still to Pose** (`PSS_PadStillToPose`) | `image` (the side still, figure on plain white), `cx`, `y0`, `y1` (centre x, top and bottom of the figure on the 768 canvas), `keep_size` (no resampling: a bigger square canvas instead), `mode` (`manual`, or `auto from pose frame 0` with `pose_frames` connected and `head_margin` / `feet_margin`). | `padded` (IMAGE), `info` |

The render takes about 1-2 seconds per frame on a CPU (65 frames: 1-2 minutes). Conventions: OpenPose R side
(2, 3, 4, 8, 9, 10) = the near side, toward the camera; the character faces right.

## The three example moves

The settings that made the example renders (the inputs are in the workflow bundle: `inputs/tracks/` and
`inputs/scail_cop/`; copy both into `ComfyUI/input/`):

| Move | Fit: canvas / hip_x / fixed_shift / pad_top | Render: crop_h / crop_off | Pad: still, cx / y0 / y1 / keep_size |
|---|---|---|---|
| short uppercut | 1024 / 347 / false / 0 | 0 / -1 | `short_uppercut_side_raw.png`, 289 / 40 / 691 / false |
| walk | 1024 / 347 / **true** / 0 | 0 / -1 | `walk_side_raw.png`, 249 / 28 / 679 / false |
| heavy uppercut | 1240 / 420 / false / 269 | 768 / 0 | `short_uppercut_side_raw.png`, 284 / 200 / 737 / **true** |

All other inputs at their defaults; the character keypoints are `scail_cop/cop/cop_keypoints.json`. The heavy uppercut
reuses the guard still of the short uppercut. For your own character read `cx`, `y0`, `y1` off your pose frame 0: `cx` =
the centre of the cylinders' x range, `y0` = a little above the head cylinder (the head), `y1` = a little below the
ankles (the boots). The `auto` mode does that with `head_margin` 0.067 and `feet_margin` 0.022 of the box height, which
gives the numbers above for the example character; check it for yours.

## Example workflows

Two workflows, because a move is extracted once and fitted to every character:

- `1_extract_poses_from_video.json` (**once per move**): Load Video (VideoHelperSuite) -> Scan Video Poses -> Repair
  Pose Scan -> Track from Pose Scan -> **Save Skeleton Track** (`ComfyUI/output/pss_tracks/<name>_vN/`), with the scan
  overlay and the track preview as check videos and the scan saved for re-cutting. Example input:
  `example_inputs/sprint_kicks_side_720p.mp4` (CC BY 3.0, see `example_inputs/ATTRIBUTION.md`): copy it into
  `ComfyUI/input/` or upload it in Load Video. About 1 minute on a CPU.
- `2_fit_track_to_character.json` (**once per character**): **Load Skeleton Track** (the folder workflow 1 printed,
  e.g. `pss_tracks/sprint_kicks_v1`) + Load Image -> DWPose Estimator (comfyui_controlnet_aux) -> Fit Track to
  Character -> Render SCAIL Cylinders -> pose video (mp4) + pose frames. For another character change only the picture.
  The example picture is `cop_on_white.png` of the Posed Sprite Sheet Generator V2 bundle. About 1-2 minutes on a CPU.

**The character picture:** 768 x 1024, the character side view facing right, whole body, on white. The fit takes the
figure's height and feet line from where it stands in this picture, and its bone lengths from its DWPose keypoints.

- `minimal_workflow.json`: the bare minimum, no models: Load Skeleton Track (the fitted short uppercut) -> Render SCAIL
  Cylinders -> Save Image, and Load Image (its side still) -> Pad Still to Pose -> Save Image.
- `example_workflow.json`: Load Skeleton Track (source) -> Fit Track to Character -> Render SCAIL Cylinders -> Video
  Combine (needs ComfyUI-VideoHelperSuite) + Save Image, and Load Image -> Pad Still to Pose -> Save Image.

Both read the tracks from `ComfyUI/input/tracks/` (copy the bundle's `inputs/tracks/` there; the example also needs
`inputs/scail_cop/` for the keypoints json) and the side still by its plain name `short_uppercut_side_raw.png`: load
it with Load Image's "choose file to upload" (Load Image only lists files at the top of the input folder).

The pose frames and the padded still are the pose video and the reference image of the SCAIL-Preview workflows of the
Civitai bundle.

## Install

- ComfyUI Manager: Custom Nodes Manager, search "PoseCapture" or "Posed Sprite Sheet", Install, restart. Or load a workflow and use
  "Install Missing Custom Nodes".
- CLI: `comfy node install PoseCapture` (or `comfy node registry-install PoseCapture`).
- Manual: `git clone https://github.com/Eloque/PoseCapture` into `ComfyUI/custom_nodes/`, restart. No pip step: numpy and Pillow ship with
  ComfyUI.
- Scan Video Poses also needs **comfyui_controlnet_aux** (ComfyUI Manager: "ComfyUI's ControlNet Auxiliary
  Preprocessors"), for DWPose; it downloads the two DWPose models (yolox_l, dw-ll_ucoco_384) itself on first use. The
  other nodes need nothing but ComfyUI.

## Tests

`tests/test_exact.py BUNDLE_DIR OUT_DIR` runs the nodes' functions on the bundle's inputs with the settings above and
compares the results with the bundle's files (fitted keypoints, every pose frame, the padded stills); it exits 0 only
when everything is identical.
`tests/test_video_graph.py` runs the video chain through a running ComfyUI on the walk reference video (not included) and its stored
scan and compares the track and the pose frames with the bundle's walk.

## License

Copyright (C) 2026 Eloque. Licensed under the GNU General Public License v3.0 (see `LICENSE`), the same licence as
ComfyUI. The example video excerpt in `example_inputs/` is CC BY 3.0 (Wellman Wellness Training; attribution in
`example_inputs/ATTRIBUTION.md`).

## How this was made

Most of the coding in this pack was done by Anthropic's Claude models, **Claude Opus** and **Claude Fable**, working
under Eloque's direction: Eloque set the goals, the rules and the checks and judged the results.
