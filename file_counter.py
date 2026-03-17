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

class SimpleStringAccumulator:
    # Global storage will now be a dictionary of lists.
    # Each key in the dictionary will be an 'accumulator_index' (int),
    # and its value will be a list of strings for that specific accumulator.
    _global_stored_strings = {} # Changed to a dictionary

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string_input": ("STRING", {"multiline": True, "default": ""}),
                "accumulator_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "display": "number"}), # New input for index
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 99999, "step": 1, "display": "number"}),
                "output_with_newlines": ("BOOLEAN", {"default": True, "label_on": "Add Newlines", "label_off": "Join Continuously"}),
                 "verbose": ("BOOLEAN", {"default": False}),
                 "print_result": ("BOOLEAN", {"default": False}),

            },
            "optional": {
                "reset_accumulator": ("BOOLEAN", {"default": False, "hidden": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("accumulated_string_output",)
    FUNCTION = "execute"
    CATEGORY = "utilities/simple_string"

    def execute(self, string_input, accumulator_index, batch_size, output_with_newlines, verbose, print_result, reset_accumulator=False):
        
        if verbose:
           print(f"SimpleStringAccumulator: Executing. Index: {accumulator_index}, string_input (first 50 chars): '{string_input[:50]}...', batch_size: {batch_size}, output_with_newlines: {output_with_newlines}.")

        # Get or create the specific string list for this accumulator_index
        if accumulator_index not in SimpleStringAccumulator._global_stored_strings:
            SimpleStringAccumulator._global_stored_strings[accumulator_index] = []
            if verbose:
               print(f"SimpleStringAccumulator: Initialized new accumulator for index {accumulator_index}.")

        current_accumulator_list = SimpleStringAccumulator._global_stored_strings[accumulator_index]

        # Append the input string to the selected accumulator
        current_accumulator_list.append(string_input)
        current_count = len(current_accumulator_list)
        if verbose:
           print(f"SimpleStringAccumulator: Accumulator {accumulator_index} current count: {current_count}.")

        if current_count >= batch_size:
            delimiter = "\n" if output_with_newlines else ""
            
            strings_to_join = list(current_accumulator_list) # Take a copy for joining
            result_string = delimiter.join(strings_to_join)
            
            # Reset the current accumulator list if 'reset_accumulator' is True
            if reset_accumulator:
               SimpleStringAccumulator._global_stored_strings[accumulator_index] = []
               if verbose:
                  print(f"SimpleStringAccumulator: Accumulator {accumulator_index} explicitly reset after output.")
            else:
                # If not resetting, clear the specific list after batch output
                SimpleStringAccumulator._global_stored_strings[accumulator_index] = []
            
            if verbose:
               print(f"SimpleStringAccumulator: Batch from accumulator {accumulator_index} ({len(strings_to_join)} strings) formed and outputted.")
            if print_result:
               print(f"Result##########################################")
               print(result_string)
            return (result_string,)
        else:
            if verbose:
               print(f"SimpleStringAccumulator: Accumulator {accumulator_index} not yet at batch size. Current: {current_count}/{batch_size}. Returning empty string.")
            return ("",)