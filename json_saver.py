import json
import os

class JsonPrettifierSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True, "default": "{}"}),
                "save_path": ("STRING", {"default": "C:/ComfyUI/output"}),
                "save_name": ("STRING", {"default": "data"}),
                "indent": ("INT", {"default": 4, "min": 0, "max": 8}),
                "strict": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pretty_json", "final_path")
    FUNCTION = "process_json"
    CATEGORY = "utils/json"

    def process_json(self, json_text, save_path, save_name, indent, strict):
        if not save_name.lower().endswith(".json"):
            save_name += ".json"
            
        final_path = os.path.join(save_path, save_name)

        try:
            
            data = json.loads(json_text, strict=strict)
            
            pretty_string = json.dumps(data, indent=indent, ensure_ascii=False)

            os.makedirs(os.path.abspath(save_path), exist_ok=True)

            with open(final_path, "w", encoding="utf-8") as f:
                f.write(pretty_string)

            return (pretty_string, str(final_path))

        except Exception as e:
            error_msg = f"JSON Error (strict={strict}): {str(e)}"
            print(error_msg)
            return (json_text, f"FAILED: {error_msg}")


class EscapeQuotesForJson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"multiline": True, "default": "Text with \"quotes\"."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("escaped_string",)
    FUNCTION = "process_string"
    CATEGORY = "utils/json"

    def process_string(self, input_string):
        json_representation_with_outer_quotes = json.dumps(input_string)
        escaped_content_without_outer_quotes = json_representation_with_outer_quotes[1:-1]
        return (escaped_content_without_outer_quotes,)

