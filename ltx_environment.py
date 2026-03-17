from pathlib import Path
import os

def _init_env():
    # --- TORCH ------
    #os.environ["TORCH_LOGS"] = "all"
    #os.environ["TORCH_LOGS"] = "+dynamo"
    #os.environ["TORCHDYNAMO_PRINT_GUARDS"] = "1"
    #os.environ["TORCH_LOGS"] = "recompiles,graph_breaks"
    #os.environ["TORCH_PIN_MEMORY"] = "1"
    #os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    #os.environ["TORCHDYNAMO_BREAK_ON_ERROR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    #-----CUDA------------
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    #-----PYTHON----------
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.resolve())
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["USE_LIBUV"] = "0"
    #-----TRAIN OFFLINE------------
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    

# --- DYNAMIC PATH RESOLUTION ---
def _resolve_script_path(script_name):
    NODE_ROOT = Path(__file__).parent.resolve()
    LTX_REPO_ROOT = NODE_ROOT / "LTX-2"
    LTX_PACKAGES_DIR = LTX_REPO_ROOT / "packages"
    LTX_TRAINER_ROOT = LTX_PACKAGES_DIR / "ltx-trainer"
    """
    Resolves the absolute path for a given script.
    Searches primarily in the new LTX-2 2.3+ monorepo structure.
    """
    trainer_scripts = LTX_TRAINER_ROOT / "scripts"
    # Fallback paths to ensure backward compatibility
    search_paths =[
        trainer_scripts,
        NODE_ROOT / "scripts",
        NODE_ROOT
    ]
    for p in search_paths:
        candidate = p / script_name
        if candidate.exists():
            return str(candidate.resolve())
            
    raise FileNotFoundError(f"Script not found: {script_name} inside {search_paths}")
#-------ENV COPY ----------
def _get_subprocess_env():
    env = os.environ.copy()
    NODE_ROOT = Path(__file__).parent.resolve()
    LTX_PACKAGES_DIR = NODE_ROOT / "LTX-2" / "packages"
    python_paths = []
    # package list
    packages_to_add = ["ltx-core", "ltx-trainer"]
    for pkg_name in packages_to_add:
        src_path = LTX_PACKAGES_DIR / pkg_name / "src"
        if src_path.exists():
            python_paths.append(str(src_path.resolve()))
    
    # script list
    scripts_path = LTX_PACKAGES_DIR / "ltx-trainer" / "scripts"
    if scripts_path.exists():
        python_paths.append(str(scripts_path.resolve()))
    # Inyect FFmpeg
    ffmpeg_bin = NODE_ROOT / "bin" / "ffmpeg" / "bin"
    if ffmpeg_bin.exists():
        env["PATH"] = str(ffmpeg_bin) + os.pathsep + env.get("PATH", "")
    # join path
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        python_paths.append(current_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env