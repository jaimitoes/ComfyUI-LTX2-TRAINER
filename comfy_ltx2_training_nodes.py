# -*- coding: utf-8 -*-
import os, sys, subprocess, yaml, re, shutil, time, json, gc, ast, atexit
import torch
from pathlib import Path
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted, LoadedModel
from ltx_trainer.model_loader import LtxModelComponents
from .ltx_environment import _init_env, _get_subprocess_env, _resolve_script_path



_active_processes = []


_init_env()

MODULE_GROUPS = {
        "video_self_attention_module": [
            "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0"
        ],
        "video_cross_attention_module": [
            "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0"
        ],
        "video_feed_forward_module": [
            "ff.net.0.proj", "ff.net.2"
        ],
        "audio_self_attention_module": [
            "audio_attn1.to_k", "audio_attn1.to_q", "audio_attn1.to_v", "audio_attn1.to_out.0"
        ],
        "audio_cross_attention_module": [
            "audio_attn2.to_k", "audio_attn2.to_q", "audio_attn2.to_v", "audio_attn2.to_out.0"
        ],
        "audio_feed_forward_module": [ 
            "audio_ff.net.0.proj", "audio_ff.net.2"
        ],
        "video_attends_to_audio_module": [ 
            "audio_to_video_attn.to_k", "audio_to_video_attn.to_q", 
            "audio_to_video_attn.to_v", "audio_to_video_attn.to_out.0"
        ],
        "audio_attends_to_video_module": [ 
            "video_to_audio_attn.to_k", "video_to_audio_attn.to_q", 
            "video_to_audio_attn.to_v", "video_to_audio_attn.to_out.0"
        ]
}

def cleanup_processes():
    for p in _active_processes:
        if p.poll() is None:
            p.kill()

atexit.register(cleanup_processes)


class LTX2_CreateConfig:
    @classmethod
    
    
    def INPUT_TYPES(s):
        return {
            "required": {

                # --- PATHS---
                "model_path": ("STRING", {"default": "", "tooltip": "Local path of the base model to be trained."}),
                "text_encoder_path": ("STRING", {"default": "", "tooltip": "Path to the base text encoder."}),
                "preprocessed_data_root": ("STRING", {"default": "", "tooltip": "Path to the directory containing preprocessed training data."}),
                "output_dir": ("STRING", {"default": "outputs/ltx2_train", "tooltip": "Directory where results and checkpoints will be saved."}),
                "load_checkpoint": ("STRING", {"default": "null", "tooltip": "Path to an existing checkpoint to resume training. You must to set (null) to bypass it "}),
                "images": ("STRING", {"default": "null", "tooltip": "Optional: First frame images for image-to-video validation (path), If provided, must have one image per prompt, multiple images are setted up with paths separated by commas"}),
                # --- PROMPTS ---
                "prompts": ("STRING", {"default": "A video of a cat...", "tooltip": "Validation prompts to monitor training progress. \n Add new prompts by adding it in a new line"}),
                "negative_prompt": ("STRING", {"default": "worst quality, blurry", "tooltip": "Negative prompts to be used during validation. \n Add new prompts by adding it in a new line"}),
                #
                "training_mode": (["lora", "full"], {"default": "lora", "tooltip": "Training mode: LoRA (memory-efficient) or Full Fine-tuning."}),
                "load_text_encoder_in_8bit": ("BOOLEAN", {"default": False, "tooltip": "Load the text encoder in 8-bit precision."}),
                "quantization": (["int8-quanto", "int4-quanto", "int2-quanto", "fp8-quanto", "fp8uz-quanto", "null"], {"default": "null", "tooltip": "Quantization for the LTX MODEL to reduce GPU memory usage."}),
                # --- MODULES  ---
                "video_self_attention_module": ("BOOLEAN", {"default": True, "tooltip": "Video self-attention"}),
                "video_cross_attention_module": ("BOOLEAN", {"default": False, "tooltip": "Video cross-attention (to text)"}),
                "video_feed_forward_module": ("BOOLEAN", {"default": False, "tooltip": "Video feed-forward network"}),
                "audio_self_attention_module": ("BOOLEAN", {"default": True, "tooltip": "Audio self-attention"}),
                "audio_cross_attention_module": ("BOOLEAN", {"default": False, "tooltip": "Audio cross-attention (to text)"}),
                "audio_feed_forward_module": ("BOOLEAN", {"default": False, "tooltip": "Audio feed-forward network"}),
                "video_attends_to_audio_module": ("BOOLEAN", {"default": True, "tooltip": "Audio attends to video (Q from audio, K/V from video)"}),
                "audio_attends_to_video_module": ("BOOLEAN", {"default": True, "tooltip": "Video attends to audio (Q from video, K/V from audio)"}),
                # --- LORA  ---
                "lora_rank": ("INT", {"default": 32, "tooltip": "Rank (dimension) of the LoRA. Higher values capture more detail but require more VRAM."}),
                "lora_alpha": ("INT", {"default": 32, "tooltip": "LoRA scaling factor. Usually set to the same value as rank or double."}),
                "lora_dropout": ("FLOAT", {"default": 0.0, "step": 0.01, "tooltip": "Dropout rate to prevent overfitting during LoRA training."}),
                
                # --- STRATEGY ---
                "strategy_name": (["text_to_video", "video_to_video"], {"default": "text_to_video", "tooltip": "Text to video for (av attention t2v-i2v). \n Video to video (Caution: Activate all video modules, Deactivate all audio modules). For CONTROL LORA (DEPTH, CANNY, ETC)"}),
                "stg_mode": (["stg_av", "stg_v"], {"default": "stg_av", "tooltip": "Spatial Temporal Guidance (STG) mode, stg_av for audio and video training, stg_v for video training."}),
                "first_frame_conditioning_p": ("FLOAT", {"default": 0.50, "step": 0.01, "tooltip": "Probability of using the first frame as conditioning during training. 0.5 recomended"}),
                # --- OPTIMIZATION ---
                "learning_rate": ("FLOAT", {"default": 1e-4, "step": 0.000001, "tooltip": "From Ltx documentation, ranges are between 1e-5 to 1e-3 for the learning curve"}),
                "steps": ("INT", {"default": 2000, "min": 250, "max": 9999, "tooltip": "Total number of training steps."}),
                "batch_size": ("INT", {"default": 1, "tooltip": "Number of samples processed in parallel per step."}),
                "gradient_accumulation": ("INT", {"default": 1}),
                "optimizer_type": (["adamw", "adamw8bit"], {"default": "adamw8bit", "tooltip": "Optimizer to use. 8-bit versions save significant memory."}),
                "mixed_precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Numerical precision. BF16 is recommended for modern GPUs."}),
                "enable_gradient_checkpointing": ("BOOLEAN", {"default": True, "tooltip": "Reduces memory usage by recomputing activations during the backward pass."}),
                "video_dims": ("STRING", {"default": "576,576,89", "tooltip": "Video dimensions (width, height, frames) separated by commas. Note: the value setted for frames will trim longer videos, shorter videos will be excluded, pay attention to this. \n **Number of frames** must satisfy `frames % 8 == 1` (e.g., 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121, etc.)"}),
                "interval": ("INT", {"default": 100, "min": 25, "max": 9999, "tooltip": "Step interval at which to perform validation. You can deactivate it by setting it to 0"}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "step": 0.1, "tooltip": "CFG scale for validation video generation."}),
                "stg_scale": ("FLOAT", {"default": 1.0, "step": 0.01, "tooltip": "STG scale applied during validation. 1.0 Recomended"}),
                "inference_steps": ("INT", {"default": 30, "tooltip": "Sampling steps used for validation inference."}),
                "frame_rate": ("FLOAT", {"default": 25.0, "step": 1.0, "tooltip": "Frames per second for generated videos."}),
                "checkpoint_interval": ("INT", {"default": 250, "min": 25, "max": 9999, "tooltip": "Steps interval at which to save model checkpoints."}),
                "checkpoints_keep_last_n": ("INT", {"default": 3, "tooltip": "Number of recent checkpoints to retain, deleting older ones."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999999999999, "tooltip": "Random seed for reproducibility."}),
                "videos_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10}),
                "num_dataloader_workers": ("INT", {"default": 2, "tooltip": "Number of subprocesses for data loading (0 is often more stable on Windows)."}),
                # ---VALIDATION ---
                "skip_initial_validation": ("BOOLEAN", {"default": False, "tooltip": "If True, skips the validation process at the very beginning of training."}),
                "reference_videos": ("STRING", {"default": "null", "tooltip": "for videe_to_video Reference validation (path), If provided, must have one video per prompt, multiple videos are setted up with paths separated by commas"}),
                "reference_downscale_factor": ("INT", {"default": 1, "min": 0, "max": 4, "tooltip": "Set to match preprocessing for reference videos (e.g., 2 for half resolution)"}),
                "include_reference_in_output": ("BOOLEAN", {"default": True, "tooltip": "video validation concatenate reference videos with generation for comparision"}),
                "scheduler_type": (["linear", "constant", "cosine", "cosine_with_restarts", "polynomial"], {"default": "linear"}),
                "timestep_sampling_mode":(["shifted_logit_normal", "uniform"], {"default": "shifted_logit_normal"}),
                
                # --- EXTRA PARAMS ---
                "scheduler_params":  ("STRING", {"multiline": True, "default": "{}", "tooltip": "Specific parameters for the scheduler in JSON/YAML format."}),
                "timestep_sampling_params": ("STRING", {"multiline": True, "default": "{}", "tooltip": "Specific parameters for the scheduler in JSON/YAML format."}),
                "extra_yaml_params": ("STRING", {"multiline": True, "default": "{}", "tooltip": "Additional parameters in YAML format for advanced configuration."}),
                
            }
        }

    RETURN_TYPES = ("STRING", "*", "BOOLEAN")
    RETURN_NAMES = ("config_path", "ltx_config", "trigger")
    FUNCTION = "generate"
    CATEGORY = "LTX-2 / Trainer"

    def validate_image_path_in_dict(self, image_path_string):
        
        if not isinstance(image_path_string, str):
           
            if image_path_string is None:
                return None
            else:
                print(f"BYPASSING INITIAL IMAGE (type: '{type(image_path_string)}'). CONVERTING TO NONE.")
                return None

        stripped_path = image_path_string.strip()

        
        if not stripped_path or stripped_path.lower() == "null":
            print(f"DEBUG: INITIAL IMAGE PATH EMPTY '{image_path_string}' BYPASSING IT.")
            return None
        
        
        if os.path.exists(stripped_path):
            return stripped_path 
        else:
            print(f"DEBUG: INITIAL IMAGE, NO FILE FOUND, BYPASSING: '{image_path_string}'.")
            return None

    def generate(self, **kwargs):
        
        
        target_modules = []
        for module_name, layers in MODULE_GROUPS.items():
            if kwargs.get(module_name):
                target_modules.extend(layers)


       
        with_audio = (kwargs["stg_mode"] == "stg_av")
        if kwargs["strategy_name"] == "text_to_video":
            strategy_cfg = {
                "name": "text_to_video",
                "first_frame_conditioning_p": kwargs["first_frame_conditioning_p"],
                "with_audio": with_audio,
                "audio_latents_dir": "audio_latents"
            }
        else:
            strategy_cfg = {
                "name": "video_to_video",
                "first_frame_conditioning_p": kwargs["first_frame_conditioning_p"],
                "reference_latents_dir": "reference_latents"
            }

        
        config_dict = {
            "model": {
                "model_path": kwargs["model_path"],
                "text_encoder_path": kwargs["text_encoder_path"],
                "training_mode": kwargs["training_mode"],
                "load_checkpoint": None if kwargs["load_checkpoint"] in ["null", ""] else kwargs["load_checkpoint"]
            },
            "lora": {
                "rank": kwargs["lora_rank"],
                "alpha": kwargs["lora_alpha"],
                "dropout": kwargs["lora_dropout"],
                "target_modules": target_modules
            },
            "training_strategy": strategy_cfg,
            "optimization": {
                "learning_rate": kwargs["learning_rate"],
                "steps": kwargs["steps"],
                "batch_size": kwargs["batch_size"],
                "gradient_accumulation_steps": kwargs["gradient_accumulation"],
                "optimizer_type": kwargs["optimizer_type"],
                "scheduler_type": kwargs["scheduler_type"],
                "enable_gradient_checkpointing": kwargs["enable_gradient_checkpointing"],
            },
            "acceleration": {
                "mixed_precision_mode": kwargs["mixed_precision"],
                "quantization": None if kwargs["quantization"] == "null" else kwargs["quantization"],
                "load_text_encoder_in_8bit": kwargs["load_text_encoder_in_8bit"]
                
            },
            "data": {
                "preprocessed_data_root": kwargs["preprocessed_data_root"],
                "num_dataloader_workers": kwargs["num_dataloader_workers"]
            },
            "validation": {
                "prompts": [p.strip() for p in kwargs["prompts"].split("\n") if p.strip()],
                "negative_prompt": kwargs["negative_prompt"],
                "images": [str(d.strip()) for d in kwargs["images"].split(",")],
                "reference_videos": [str(d.strip()) for d in kwargs["reference_videos"].split(",")],
                "video_dims": [int(d.strip()) for d in kwargs["video_dims"].split(",")],
                "frame_rate": kwargs["frame_rate"],
                "seed": kwargs["seed"],
                "inference_steps": kwargs["inference_steps"],
                "interval": kwargs["interval"] if kwargs["interval"] > 0 else "null",
                "guidance_scale": kwargs["guidance_scale"],
                "stg_scale": kwargs["stg_scale"],
                "stg_blocks": [29],
                "stg_mode": kwargs["stg_mode"],
                "generate_audio": with_audio,
                "skip_initial_validation": kwargs["skip_initial_validation"],
                "videos_per_prompt": kwargs["videos_per_prompt"]
                
            },
            "checkpoints": {
                "interval": kwargs["checkpoint_interval"],
                "keep_last_n": kwargs["checkpoints_keep_last_n"],
                "precision": "bfloat16"
            },
            "flow_matching": {
                "timestep_sampling_mode": kwargs["timestep_sampling_mode"], 
                "timestep_sampling_params": {} 
            },
            "output_dir": kwargs["output_dir"],
            "seed": kwargs["seed"],
            "wandb": {"enabled": False},
            "hub": {"push_to_hub": False}
        }
        
        
        validated_image_paths = []
        for img_str in config_dict["validation"]["images"]:
            valid_path = self.validate_image_path_in_dict(img_str)
            if valid_path is not None:
                validated_image_paths.append(valid_path)
        config_dict["validation"]["images"] = validated_image_paths if validated_image_paths else None

        if kwargs["strategy_name"] == "video_to_video":
            config_dict["validation"]["reference_downscale_factor"] = kwargs["reference_downscale_factor"]
            config_dict["validation"]["include_reference_in_output"] = kwargs["include_reference_in_output"]
            validated_video_paths = []
            print(f"Strategy VIDEO TO VIDEO, GETTING PATH : {str(validated_video_path)}")
            for vid_str in config_dict["validation"]["reference_videos"]:
                valid_video_path = self.validate_image_path_in_dict(vid_str)
                if valid_video_path is not None:
                    validated_video_paths.append(valid_video_path)
            config_dict["validation"]["reference_videos"] = validated_video_paths if validated_video_paths else None
        else:
            del config_dict["validation"]["reference_videos"]
       
        if kwargs.get("scheduler_params") and kwargs["scheduler_params"].strip() not in ["{}", ""]:
            config_dict["optimization"]["scheduler_params"] = json.loads(kwargs["scheduler_params"].replace("'", '"'))
   

   
        if kwargs.get("timestep_sampling_params") and kwargs["timestep_sampling_params"].strip() not in ["{}", ""]:
            config_dict["flow_matching"]["timestep_sampling_params"] = json.loads(kwargs["timestep_sampling_params"].replace("'", '"'))
                
        
        out_folder = Path(kwargs["output_dir"]) / "configs"
        out_folder.mkdir(parents=True, exist_ok=True)
        config_file = out_folder / f"config_{int(time.time())}.yaml"
        
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        ltx_config = config_dict.copy()
        return (str(config_file.resolve()), ltx_config, True)


class LTX2_SceneSplitter:
    """
    ComfyUI node for splitting video into scenes using an external script.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
                "trigger": ("BOOLEAN", {"default": True, "tooltip": "Trigger the node execution.", "forceInput": True}),
                # --------------------------------------------------
                "video_path": ("STRING", {"default": "", "tooltip": "Path to the input video file."}),
                "output_directory": ("STRING", {"default": "", "tooltip": "Directory to save output files (e.g., images). A subdirectory will be created for each video based on its name."}),
                "detector": (["content", "adaptive", "threshold", "histogram"], {"default": "content", "tooltip": "Scene detection algorithm to use."}),
                "threshold": ("FLOAT", {"default": 27.0, "min": 0.0, "max": 1000.0, "step": 0.1, "tooltip": "Detection threshold (meaning varies by detector)."}),
                "max_scenes": ("INT", {"default": 30, "min": 0, "max": 999999, "tooltip": "Maximum number of scenes to produce (0 for unlimited)."}),
                "min_scene_length": ("STRING", {"default": "49", "tooltip": "Minimum scene length during detection, in frames. Forces detector to make scenes at least this many frames."}),
                "filter_shorter_than": ("STRING", {"default": "49", "tooltip": "Filter out scenes shorter than this duration in frames"}),
                
                "skip_start": ("STRING", {"default": "0", "tooltip": "Number of frames (e.g., '100'), seconds (e.g., '3s'), or timecode (e.g., '00:00:05.000') to skip at the start of the video. '0' or empty to not skip."}),
                "skip_end": ("STRING", {"default": "0", "tooltip": "Number of frames (e.g., '100'), seconds (e.g., '3s'), or timecode (e.g., '00:00:05.000') to skip at the end of the video. '0' or empty to not skip."}),
                "duration_process": ("STRING", {"default": "0", "tooltip": "How much of the video to process. Can be frames (e.g., '100'), seconds (e.g., '3s'), or timecode (e.g., '00:00:05.000'). Empty or '0' to process the entire video. Note: The underlying script may interpret 'duration' as an end time or total length from start, depending on other skip values."}),
                # ------------------------------------------------------------------
                "save_images_per_scene": ("INT", {"default": 0, "min": 0, "max": 10, "tooltip": "Number of preview images to save per scene (0 to disable). These are saved in the scene output directory."}),
                # REMOVED: "stats_file_path" input
                "luma_only": ("BOOLEAN", {"default": False, "tooltip": "Only use brightness for content detection."}),
                "adaptive_window": ("INT", {"default": 0, "min": 0, "tooltip": "Window size for adaptive detection (0 uses detector default)."}),
                "fade_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Bias for fade detection (-1.0 to 1.0)."}),
                "downscale_factor": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "Factor to downscale frames by during detection (e.g., 2 for half resolution)."}),
                "frame_skip_process": ("INT", {"default": 0, "min": 0, "tooltip": "Number of frames to skip during processing (0 processes all frames for full accuracy)."}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", )
    RETURN_NAMES = ("directory_path", "trigger", )
    FUNCTION = "split_scenes_execute"
    CATEGORY = "LTX-2 / Video Processing"

    def split_scenes_execute(self,
                             trigger: bool,
                             video_path: str,
                             output_directory: str,
                             detector: str,
                             threshold: float,
                             max_scenes: int,
                             min_scene_length: str,
                             filter_shorter_than: str,
                             skip_start: str,    # Type hint changed to str
                             skip_end: str,      # Type hint changed to str
                             duration_process: str, # Type hint changed to str
                             save_images_per_scene: int,
                             luma_only: bool,
                             adaptive_window: int,
                             fade_bias: float,
                             downscale_factor: int,
                             frame_skip_process: int):

        if not trigger:
            print("LTX2 SceneSplitter: Trigger is False, skipping execution.")
            return (output_directory, False)

        # 1. Input Validation
        if not video_path:
            raise RuntimeError("LTX2 SceneSplitter Error: 'video_path' cannot be empty.")
        video_file_path = Path(video_path).resolve()
        if not video_file_path.is_file():
            raise RuntimeError(f"LTX2 SceneSplitter Error: Video file not found: {video_file_path}")

        # Create a unique output subdirectory for this video based on its name
        final_output_dir = Path(output_directory).resolve()
        final_output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Resolve Script Path
        script = _resolve_script_path("split_scenes.py")

        # 3. Construct Command
        cmd = [
            sys.executable, "-u", str(script),
            str(video_file_path),                 # First positional argument: VIDEO_PATH
            str(final_output_dir),                # Second positional argument: OUTPUT_DIR
        ]

        # Add command-line arguments based on inputs
        if detector != "content":
            cmd.extend(["--detector", detector])
        
        cmd.extend(["--threshold", str(threshold)])

        if max_scenes > 0:
            cmd.extend(["--max-scenes", str(max_scenes)])

        if min_scene_length and str(min_scene_length).strip().isdigit():
            if int(min_scene_length) > 0:
                cmd.extend(["--min-scene-length", str(min_scene_length)])
 
        if filter_shorter_than and str(filter_shorter_than).strip().isdigit(): 
                cmd.extend(["--filter-shorter-than", str(filter_shorter_than)])

        # --- MODIFICATION: Handle timing arguments as strings ---
        # Trim whitespace from user input strings
        clean_skip_start = skip_start.strip()
        clean_skip_end = skip_end.strip()
        clean_duration_process = duration_process.strip()

        # Only add if non-empty and not "0" (which means no skip/no effect)
        if clean_skip_start and clean_skip_start != "0":
            cmd.extend(["--skip-start", clean_skip_start])

        if clean_skip_end and clean_skip_end != "0":
            cmd.extend(["--skip-end", clean_skip_end])

        # REMOVED: The --duration argument, as split_scenes.py reported "No such option".
        # If duration control is critical, the split_scenes.py script itself would need to be updated
        # to support an explicit duration flag, or --end would need to be calculated based on --skip-start + duration.
        # For now, if duration_process is used, it will not be passed to the external script.
        # --------------------------------------------------------

        if save_images_per_scene > 0:
            cmd.extend(["--save-images", str(save_images_per_scene)])
            
        # REMOVED: Logic for stats_file_path

        if luma_only:
            cmd.append("--luma-only")

        if adaptive_window > 0:
            cmd.extend(["--adaptive-window", str(adaptive_window)])

        cmd.extend(["--fade-bias", str(fade_bias)])

        if downscale_factor > 1:
            cmd.extend(["--downscale", str(downscale_factor)])

        if frame_skip_process > 0:
            cmd.extend(["--frame-skip", str(frame_skip_process)])
        
        print(f"LTX2 SceneSplitter: Executing command: {' '.join(cmd)}")

        # 4. Execute Subprocess
        process = None
        try:
            pbar = comfy.utils.ProgressBar(100)
            current_env = _get_subprocess_env()

            
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout for easier logging
                text=True,
                encoding='utf-8',
                errors='replace',
                env=current_env,
                cwd=str(final_output_dir), # Run subprocess in the designated output directory
                bufsize=1 # Line-buffered output
            )
            _active_processes.append(process)

            while True:
                throw_exception_if_processing_interrupted()

                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    line_str = line.strip()
                    print(f"LTX2 SceneSplitter: {line_str}")
                    # Basic progress parsing (adapt if split_scenes.py has a specific format for progress)
                    match = re.search(r'(?:Processing|Analyzing)\s+frame\s+(\d+)\s+of\s+(\d+)', line_str, re.IGNORECASE)
                    if match:
                        current_frame = int(match.group(1))
                        total_frames = int(match.group(2))
                        if total_frames > 0:
                            pbar.update_absolute(current_frame, total_frames)
                    sys.stdout.flush()

            if process.wait() != 0:
                raise RuntimeError(f"LTX2 SceneSplitter FAILED. Process exited with code {process.returncode}. Check console for details.")

        except Exception as e:
            print(f"LTX2 SceneSplitter: Process interrupted or failed: {e}")
            if process and process.poll() is None:
                _active_processes.remove(process)
                process.kill()
                process.wait()
            raise e
        finally:
            if process and process in _active_processes: 
                _active_processes.remove(process)
            if process and process.poll() is None: 
                process.kill()
                process.wait()
        
        print(f"LTX2 SceneSplitter: Scene splitting completed. Outputs in: {final_output_dir}")
        return (str(final_output_dir), trigger,)
class LTX2_AutoCaptioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_directory": ("STRING", {"default": ""}),
                "dataset_name": ("STRING", {"default": "dataset_name"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["8-bit", "full"], {"default": "8-bit"}),
                "fps": ("INT", {"default": 1, "min": 1}),
                "with_audio": ("BOOLEAN", {"default": True}),
                "clean_context": (["clean_caption", "raw_caption"], {"default": "clean_caption", "tooltip": "Whether to clean up captions by removing common VLM patterns"}),
                "skip_if_exists": ("BOOLEAN", {"default": True}),
                "instruction": ("STRING", {"multiline": True, "default": "Analyze this media and provide a detailed caption in the following EXACT format. Fill in ALL sections:\n[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>\n[SPEECH]: <Word-for-word transcription of everything spoken.\n           Listen carefully and transcribe the exact words. If no speech, write ""None"">\n[SOUNDS]: <Description of music, ambient sounds, sound effects. If none, write ""None"">\n[TEXT]: <Any on-screen text visible. If none, write ""None"">\nYou MUST fill in all four sections. For [SPEECH], transcribe the actual words spoken, not a summary.", "tooltip": "If you keep blank, it will take the default video or video-audio depending of your setting to set the english instruction"}),
                "cuda_visible_devices": ("STRING", {"default": "0", "tooltip": "Make visible your gpus by setting it per commas, example: 0 or 0,1,2"}),
            }  
                
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("pretty_json", "trigger")
    FUNCTION = "execute"
    CATEGORY = "LTX-2 / Dataset"

    def execute(self, video_directory, dataset_name, device, precision, fps, with_audio, clean_context, skip_if_exists, instruction, cuda_visible_devices):
        if not video_directory or not os.path.exists(video_directory):
            raise RuntimeError(f"LTX-2 Caption Error: Video directory not found: {video_directory}")
        output_filename = dataset_name + ".json"
        current_env = _get_subprocess_env()
        current_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(f"CURRENT CUDA GPUS ARE : {cuda_visible_devices}")
        video_dir_path = Path(video_directory).resolve()
        output_path = video_dir_path / output_filename if not os.path.isabs(output_filename) else Path(output_filename)
        
        if not (skip_if_exists and output_path.exists()):
            script = _resolve_script_path("caption_videos.py")
            
            cmd = [
                sys.executable, "-u", script, 
                str(video_dir_path), 
                "--output", str(output_path), 
                "--device", device,
                "--fps", str(fps), 
                
               
            ]

            if clean_context == "clean_caption":
                cmd.append("--clean-caption")
            elif clean_context == "raw_caption":
                cmd.append("--raw-caption")

            if instruction != "": cmd.extend(["--instruction", instruction])
            if precision == "8-bit": cmd.append("--use-8bit")
            if not with_audio: cmd.append("--no-audio")

            process = None
            try:
                pbar = comfy.utils.ProgressBar(100)
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    encoding='utf-8', 
                    errors='replace', 
                    env=current_env, 
                    cwd=str(video_dir_path), 
                    bufsize=1
                )
                _active_processes.append(process)
                while True:
                    
                    throw_exception_if_processing_interrupted()
                    
                    line = process.stdout.readline()
                    if not line and process.poll() is not None: 
                        break
                        
                    if line: 
                        
                        line_str = line.strip()
                        print(f"CAPTIONER: {line.strip()}")
                        if "CAPTIONER" in line_str and "/" in line_str:
                            try:
                                match = re.search(r'CAPTIONER (\d+)/(\d+)', line_str)
                                if match:
                                    pbar.update_absolute(int(match.group(1)), int(match.group(2)))
                            except Exception: 
                                pass
                        sys.stdout.flush()
                        
                
                if process.wait() != 0:
                    raise RuntimeError("LTX-2 AutoCaption FAILED. Check console.")
            
            except Exception as e:
                print(f"LTX2 Training: Captioning Process interrupted: {e}")
                
                if process and process.poll() is None:
                    _active_processes.remove(process)
                    process.kill()
                    process.wait()
                raise e
            
            finally:
                if process and process.poll() is None:
                    _active_processes.remove(process)
                    process.kill()
                    process.wait()
        # --- Return Json ---
        json_content = ""
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    json_content = f.read()
            except Exception as e:
                print(f"LTX2 AutoCaption Error al leer el JSON: {e}")
        # -----------------------------------
        return (json_content, True,)
    

class LTX2_RunPreprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("BOOLEAN", {"forceInput": True}),
                "config_path": ("STRING", {"forceInput": True}),
                "ltx_config": ("*", {"forceInput": True}),
                "dataset_override": ("STRING", {"forceInput": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
                "vae_tiling": ("BOOLEAN", {"default": True}),
                "load_encoder_in_8bit": ("BOOLEAN", {"default": False}), 
                "decode_preprocess": ("BOOLEAN", {"default": False, "tooltip": "This allows you to visually and audibly inspect the processed data. Other files will skip the process if exists but decode always happend so, \n if you will use existing dataset, you can disable it in the next running."}),
                "reference_path": ("STRING", {"default": "", "tooltip": " IC lora mode must be seteed. Refernce videos as Depth or Pose must be included in the directory and setted up in the Json file as \"/\"reference_path\"/\": \"/\"cat_playing_depth.mp4\"/\""}),
                "lora_trigger": ("STRING", {"default": "My_lora_trigger"}),
                "cuda_visible_devices": ("STRING", {"default": "0", "tooltip": "Make visible your gpus by setting it per commas, example: 0 or 0,1,2"}),
                
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "execute"
    CATEGORY = "LTX-2 / Dataset"

    def execute(self, trigger, config_path, ltx_config, dataset_override, batch_size, device, vae_tiling, load_encoder_in_8bit, decode_preprocess, reference_path, lora_trigger, cuda_visible_devices):
        
        if not trigger:
            raise RuntimeError("LTX-2 Preprocess HALTED: Previous Captioning node failed or was skipped incorrectly.")
        
        dataset_file = dataset_override if dataset_override else config_path
        print(f"DEBUG: LTX-2 Preprocess intentando abrir el archivo: {dataset_file}")
        if not os.path.exists(dataset_file):
            raise RuntimeError(f"LTX-2 Preprocess Error: Dataset file not found: {dataset_file}")
        current_env = _get_subprocess_env()
        current_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(f"CURRENT CUDA GPUS ARE : {cuda_visible_devices}")
        pre_dir = ltx_config['data']['preprocessed_data_root']
        script = _resolve_script_path("process_dataset.py")
        

        cmd = [
            sys.executable, "-u", script, str(dataset_file),
            "--resolution-buckets", f"{ltx_config['validation']['video_dims'][0]}x{ltx_config['validation']['video_dims'][1]}x{ltx_config['validation']['video_dims'][2]}",
            "--model-path", ltx_config['model']['model_path'],
            "--text-encoder-path", ltx_config['model']['text_encoder_path'],
            "--output-dir", pre_dir, "--batch-size", str(batch_size), "--device", device
        ]
        
        if lora_trigger.strip():
            cmd.extend(["--lora-trigger", lora_trigger.strip()])
        else:
            raise RuntimeError("Error! Avoiding a disappointment! A Lora trigger has not been assigned on the node.")
        if load_encoder_in_8bit:
            cmd.append("--load-text-encoder-in-8bit")
       
       
        if ltx_config['validation']['generate_audio']: cmd.append("--with-audio")
        if vae_tiling: cmd.append("--vae-tiling")
        if decode_preprocess: cmd.append("--decode")
        if ltx_config['training_strategy']["name"] == "video_to_video": cmd.extend(["--reference_column", reference_path])
            


        print(f"LTX-2 Preprocess: Running...")
        process = None
        try:
            pbar = comfy.utils.ProgressBar(100)
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding='utf-8', 
                errors='replace', 
                env=current_env, 
                cwd=str(Path(script).parent), 
                bufsize=1
            )
            _active_processes.append(process)
            while True:
                
                throw_exception_if_processing_interrupted()
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                    
                if line:
                    line_str = line.strip()
                    print(f"PREPROCESS: {line_str}")
                    if "PREPROCESS" in line_str and "/" in line_str:
                        try:
                            match = re.search(r'PREPROCESS (\d+)/(\d+)', line_str)
                            if match:
                                pbar.update_absolute(int(match.group(1)), int(match.group(2)))
                        except Exception: 
                            pass
                    sys.stdout.flush()
        
            if process.wait() != 0:
                raise RuntimeError("LTX-2 Preprocess FAILED.")
        
            return (True,)
        
        except Exception as e:
            
            print(f"LTX2 Training: Process interrupted: {e}")
            if process and process.poll() is None:
                    
                    _active_processes.remove(process)
                    process.kill()
                    process.wait()
            raise e
        
        finally:
            if process and process.poll() is None:
                   
                    _active_processes.remove(process)
                    process.kill()
                    process.wait()


class LTX2_RunTraining:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger": ("BOOLEAN", {"forceInput": True}),
                "config_path": ("STRING", {"forceInput": True}),
                "ltx_config": ("*", {"forceInput": True}),
                "num_processes": ("INT", {"default": 1}),
                "num_cpu_threads_per_process": ("INT", {"default": 1}),
                "mixed_precision": (["bf16", "fp16", "fp8", "no"], {"default": "bf16"}),
                "attention": (["xformers", "flash_attention_3", "none"], {"default": "xformers"}),
                "use_accelerate": ("BOOLEAN", {"default": True}),
                "dynamo_backend": (["inductor", "eager", "no"], {"default" : "inductor"}),
                "dynamo_cache_size_limit": ("INT", {"default" : 64, "min": 8, "max": 4096, "tooltip": " 64 seems to be enough after recompile fixes. \n it recompiles 2 times at the begining for validation and train and nevermore break. But feel free to test other values"}),
                "disable_progress_bars": ("BOOLEAN", {"default": True}),
                "cuda_visible_devices": ("STRING", {"default": "0", "tooltip": "Make visible your gpus by setting it per commas, example: 0 or 0,1,2"}),
                "fsdp_path": ("STRING", {"default": "", "tooltip": "Accelerate launch : Enable fsdp_multiGPU by giving a path"}),
                "extra_args": ("STRING", {"multiline": True, "default": "{}"}),
            }
        }
        
    RETURN_TYPES = ("STRING",); RETURN_NAMES = ("status",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "LTX-2 / Trainer"

    def execute(self, trigger, config_path, ltx_config, num_processes, num_cpu_threads_per_process, mixed_precision, attention, use_accelerate, dynamo_backend, dynamo_cache_size_limit, disable_progress_bars,  cuda_visible_devices, fsdp_path, extra_args):
        # 1. Basic validation
        if "error" in str(trigger).lower(): 
            return ("Skipped",)
       
        # ACCELERATE
        script = _resolve_script_path("run_ltx_training.py")
        num_machines = 1
       
        cmd = []
        if use_accelerate:
            
            cmd.extend(
                [
                shutil.which("accelerate") or "accelerate", "launch"
                ]
            )

            if fsdp_path and fsdp_path.strip():
                resolved_fsdp_path = Path(fsdp_path).resolve()
                print(f"fsdp path is : {resolved_fsdp_path}")
                if not resolved_fsdp_path.is_file():
                    raise RuntimeError(f"LTX2 Training Error: FSDP path not found: '{resolved_fsdp_path}'.")
                cmd.extend(["--config_file", str(resolved_fsdp_path)])

            cmd.extend(
                [
                "--num_machines", str(num_machines),
                "--num_processes", str(num_processes), 
                "--num_cpu_threads_per_process", str(num_cpu_threads_per_process), 
                "--mixed_precision", mixed_precision,
                "--dynamo_backend", dynamo_backend,
                
                
                ]
            )
            try:
                extras = json.loads(extra_args.replace("'", '"'))
                for k, v in extras.items(): 
                    cmd.extend([str(k), str(v)])
            except Exception: 
                print("ACCELERATE: Accelerate extra argument ignored or malformed...")
            # 4. Add the script and the config path
            cmd.extend(
               [
               str(script), 
               "--config_path", str(config_path),
               
               
               ]
            )
        #Without accelerate
        else:
            
            cmd.extend(
                [
                sys.executable, 
                str(script), 
                "--config_path", str(config_path),
                ]
            )

        if attention != "none":
            cmd.extend(["--extra", attention])
        if disable_progress_bars:
            cmd.append("--disable_progress_bars")


        cmd.extend(
            [
                "--dynamo_cache_size_limit", str(dynamo_cache_size_limit),
            ]
        )
        
       
        
        print("\nLTX2 COMMANDS:", " ".join(cmd))
       
        custom_env = _get_subprocess_env()
        custom_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(f"CURRENT CUDA GPUS ARE : {cuda_visible_devices}")
       
        
        
        
        process = None 
        try:
            pbar = comfy.utils.ProgressBar(100)
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding='utf-8', 
                errors='replace', 
                env=custom_env, 
                cwd=str(Path(script).parent),
                bufsize=1
            )
            _active_processes.append(process)
            while True:
                
                throw_exception_if_processing_interrupted()
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                if line:
                    line_str = line.strip()
                    print(f"TRAIN: {line_str}")
                    
                    if "Step" in line_str and "/" in line_str:
                        try:
                            match = re.search(r'Step (\d+)/(\d+)', line_str)
                            if match:
                                pbar.update_absolute(int(match.group(1)), int(match.group(2)))
                        except Exception: 
                            pass
                    sys.stdout.flush()
        
            return_code = process.wait()
            return ("Success" if return_code == 0 else f"Failed (Code {return_code})",)
        
        except Exception as e:
            
            print(f"LTX2 Training: Process interrupted: {e}")
            if process and process.poll() is None:
                    
                    _active_processes.remove(process)
                    process.kill()
                    process.wait()
            raise e
            
        finally:
            if process and process.poll() is None:
                    
                    _active_processes.remove(process)
                    process.kill()
                    process.wait()