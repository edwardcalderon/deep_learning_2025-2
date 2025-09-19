import json
import os
from pathlib import Path

def fix_json_file(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Create a new dictionary to store the fixed data
    fixed_data = {}
    
    # Get the directory containing the JSON file
    json_dir = os.path.dirname(json_file_path)
    
    # Get list of actual image files in the directory
    image_files = [f for f in os.listdir(json_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create a set of valid filenames for quick lookup
    valid_filenames = set(image_files)
    
    # Fix the JSON data
    fixed_count = 0
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
            
        # Get the original filename from the value
        original_filename = value.get('filename', '')
        
        if not original_filename:
            print(f"Warning: Missing filename in entry: {key}")
            continue
        
        # Check if the filename exists as is
        if original_filename in valid_filenames:
            fixed_key = original_filename
        else:
            # Try to find a matching filename by base name
            base_name = original_filename.split('.')[0]
            matching_files = [f for f in image_files if f.startswith(base_name)]
            
            if len(matching_files) == 1:
                fixed_key = matching_files[0]
                fixed_count += 1
            else:
                print(f"Warning: Could not find unique matching file for {original_filename}")
                continue
        
        # Update the entry with the fixed key and filename
        value['filename'] = fixed_key
        fixed_data[fixed_key] = value
    
    # Write the fixed data back to the file
    with open(json_file_path, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    return fixed_count

# Fix both train and val JSON files
base_dir = Path("balloon")
train_json = base_dir / "train" / "via_region_data.json"
val_json = base_dir / "val" / "via_region_data.json"

print(f"Fixing {train_json}...")
num_fixed = fix_json_file(train_json)
print(f"Fixed {num_fixed} entries in {train_json}")

print(f"\nFixing {val_json}...")
num_fixed = fix_json_file(val_json)
print(f"Fixed {num_fixed} entries in {val_json}")

print("\nJSON files have been fixed. You can now run the notebook again.")
