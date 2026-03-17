import os, re

class PathAccumulator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pasted_paths": ("STRING", {"multiline": True, "default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "count": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "extensions": ("STRING", {"default": "png, wav, mp4"}),
            },
            "optional": {
                "folder_path": ("STRING", {"default": ""}),
            }
        }

    # He cambiado COMBO por LIST para mayor compatibilidad
    RETURN_TYPES = ("STRING", "LIST", "STRING")
    RETURN_NAMES = ("path", "path_list", "filename")
    FUNCTION = "get_path"
    CATEGORY = "utils"

    def get_path(self, pasted_paths="", index=0, count=1, extensions="", folder_path=""):
        # SEGURIDAD: Si los inputs llegan como None, los convertimos a string vacío
        pasted_paths = pasted_paths or ""
        folder_path = folder_path or ""
        extensions = extensions or ""
        
        # Procesar extensiones
        ext_list = ["." + e.strip().lower().lstrip('.') for e in re.split(r'[,\s\[\]]+', extensions) if e.strip()]
        
        items = []
        # Prioridad 1: Buscar en carpeta si el path existe
        if folder_path and os.path.isdir(folder_path):
            try:
                items = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                         if os.path.isfile(os.path.join(folder_path, f)) and 
                         any(f.lower().endswith(ex) for ex in ext_list)]
                items.sort()
            except Exception as e:
                print(f"[PathAccumulator] Error accediendo a la carpeta: {e}")
        
        # Prioridad 2: Si no hay items de carpeta, usar los del multiline
        if not items:
            items = [l.strip().strip('"').strip("'") for l in pasted_paths.splitlines() if l.strip()]

        # Si después de todo no hay nada, devolvemos strings vacíos (NUNCA None)
        if not items:
            return ("", [], "")
        
        # Calcular rangos
        start_idx = max(0, min(index, len(items) - 1))
        end_idx = min(start_idx + count, len(items))
        
        selected_range = items[start_idx:end_idx]
        
        # Resultado principal
        selected_path = selected_range[0] if selected_range else ""
        filename = os.path.basename(selected_path) if selected_path else ""
        
        return (str(selected_path), selected_range, str(filename))