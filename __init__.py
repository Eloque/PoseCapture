"""PoseCapture (Posed Sprite Sheet Generator): video -> skeleton track -> fitted track -> SCAIL cylinder pose frames.
The video scan uses comfyui_controlnet_aux's DWPose; every other node is CPU numpy + Pillow, no models."""
from .nodes import NODE_CLASS_MAPPINGS as _M1, NODE_DISPLAY_NAME_MAPPINGS as _D1
from .nodes_video import NODE_CLASS_MAPPINGS as _M2, NODE_DISPLAY_NAME_MAPPINGS as _D2
from .nodes_preview import NODE_CLASS_MAPPINGS as _M3, NODE_DISPLAY_NAME_MAPPINGS as _D3

NODE_CLASS_MAPPINGS = {**_M2, **_M3, **_M1}
NODE_DISPLAY_NAME_MAPPINGS = {**_D2, **_D3, **_D1}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
