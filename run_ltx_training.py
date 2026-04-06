# -*- coding: utf-8 -*-
import os, sys, gc
from pathlib import Path
import argparse
import yaml
import torch
## --- INTERNAL PATH INJECTION (Safety Fallback for Accelerate) ---
# Since this script now lives in the node root, we point to the cloned repo
_script_dir = Path(__file__).resolve().parent
_ltx_packages = _script_dir / "LTX-2" / "packages"

_paths_to_add =[]
if _ltx_packages.exists():
    for pkg_dir in _ltx_packages.iterdir():
        if pkg_dir.is_dir():
            src_dir = pkg_dir / "src"
            if src_dir.exists():
                _paths_to_add.append(str(src_dir.resolve()))
                
    # Also add the trainer scripts folder for internal dependencies
    _trainer_scripts = _ltx_packages / "ltx-trainer" / "scripts"
    if _trainer_scripts.exists():
        _paths_to_add.append(str(_trainer_scripts.resolve()))

for p in _paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(_script_dir))
# ----------------------------------------------------------------
from ltx_environment import _get_subprocess_env
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer
env = _get_subprocess_env()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--extra", type=str, default=None)
    parser.add_argument("--disable_progress_bars", action="store_true")
    parser.add_argument("--dynamo_cache_size_limit", type=str, default= "8")
    args = parser.parse_args()
    os.environ["DYNAMO_CACHE"] = args.dynamo_cache_size_limit
    del args.dynamo_cache_size_limit
    with open(args.config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    trainer_config = LtxTrainerConfig(**config_data)
   
    print(f"[LTX-BRIDGE] Initializing Trainer with: {args.config_path}")
    
    trainer = LtxvTrainer(trainer_config)
    print("[LTX-BRIDGE] Models loaded. Forcing VRAM cleanup before training loop...")

    # Aggressive VRAM cleanup before the first training step
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

if __name__ == "__main__":
    main()