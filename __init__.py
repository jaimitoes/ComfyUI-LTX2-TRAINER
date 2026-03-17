# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
# Resolve node root and target the cloned LTX-2 repo packages
node_root = Path(__file__).parent.resolve()
ltx_packages_dir = node_root / "LTX-2" / "packages"
# Dynamically inject all 'src' paths from the sub-packages into sys.path
paths_to_add =[]
if ltx_packages_dir.exists():
    for pkg_dir in ltx_packages_dir.iterdir():
        if pkg_dir.is_dir():
            src_dir = pkg_dir / "src"
            if src_dir.exists():
                paths_to_add.append(str(src_dir))
# Inject paths to the main ComfyUI environment
for p in paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)
# Import node classes
from .comfy_ltx2_training_nodes import (
    LTX2_SceneSplitter, LTX2_AutoCaptioning, LTX2_CreateConfig, 
    LTX2_RunPreprocess, LTX2_RunTraining
)
from .file_counter import FileCounter, SimpleStringAccumulator
from .path_accumulator import PathAccumulator
from .json_saver import JsonPrettifierSaver, EscapeQuotesForJson

NODE_CLASS_MAPPINGS = {
    "LTX2_SceneSplitter": LTX2_SceneSplitter,
    "LTX2_AutoCaptioning": LTX2_AutoCaptioning,
    "LTX2_CreateConfig": LTX2_CreateConfig,
    "LTX2_RunPreprocess": LTX2_RunPreprocess,
    "LTX2_RunTraining": LTX2_RunTraining,
    "FileCounter": FileCounter,
    "PathAccumulator" : PathAccumulator,
    "JsonPrettifierSaver": JsonPrettifierSaver,
    "EscapeQuotesForJson": EscapeQuotesForJson,
    "SimpleStringAccumulator": SimpleStringAccumulator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2_SceneSplitter": "LTX-2 Scene splitter",
    "LTX2_AutoCaptioning": "LTX-2 Auto Captioning (Local Qwen)",
    "LTX2_CreateConfig": "LTX-2 Create Config (Full Parameter Control)",
    "LTX2_RunPreprocess": "LTX-2 Run Preprocessing (Auto Pathing)",
    "LTX2_RunTraining": "LTX-2 Run Training (Accelerate)",
    "FileCounter": "File Counter",
    "PathAccumulator": "Path Accumulator",
    "JsonPrettifierSaver": "JSON Prettifier & Saver",
    "EscapeQuotesForJson": "STRING escape quotes",
    "SimpleStringAccumulator": "Simple String Accumulator"
}

__all__ =["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]