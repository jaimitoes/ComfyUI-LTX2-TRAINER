import os, re

class FileCounter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "extensions": ("STRING", {"default": "png, wav, mp4"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count_files"
    CATEGORY = "utils"

    def count_files(self, folder_path, extensions):
        if not folder_path or not os.path.isdir(folder_path):
            return (0,)
        
        ext_list = ["." + e.strip().lower().lstrip('.') for e in re.split(r'[,\s\[\]]+', extensions) if e.strip()]
        
        files = [f for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f)) and 
                 any(f.lower().endswith(ex) for ex in ext_list)]
        
        return (len(files),)