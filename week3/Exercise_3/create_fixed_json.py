import json
import os
from pathlib import Path

def create_fixed_json(json_file_path):
    # Read the original JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Create a new dictionary to store the fixed data
    fixed_data = {}
    
    # Get the directory containing the JSON file
    json_dir = os.path.dirname(json_file_path)
    
    # Get list of actual image files in the directory
    image_files = [f for f in os.listdir(json_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create a mapping from base filenames to their full paths
    filename_map = {}
    for img_file in image_files:
        # The base name is the filename without the extension
        base_name = os.path.splitext(img_file)[0]
        filename_map[base_name] = img_file
    
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
        
        # The key is in the format "filename[size]"
        # Extract the base name (without extension and size)
        base_name = os.path.splitext(original_filename)[0]
        
        # Find the matching file in the directory
        if base_name in filename_map:
            fixed_key = filename_map[base_name]
            # Update the filename in the value
            value['filename'] = fixed_key
            fixed_data[fixed_key] = value
            fixed_count += 1
        else:
            print(f"Warning: Could not find matching file for {original_filename}")
    
    # Write the fixed data to a new file
    output_file = os.path.join(json_dir, 'via_region_data_fixed.json')
    with open(output_file, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    return fixed_count, output_file

# Process both train and val JSON files
base_dir = Path("balloon")
train_json = base_dir / "train" / "via_region_data.json"
val_json = base_dir / "val" / "via_region_data.json"

print(f"Processing {train_json}...")
num_fixed, output_file = create_fixed_json(train_json)
print(f"Created fixed JSON file with {num_fixed} entries: {output_file}")

print(f"\nProcessing {val_json}...")
num_fixed, output_file = create_fixed_json(val_json)
print(f"Created fixed JSON file with {num_fixed} entries: {output_file}")

print("\nTo use the fixed JSON files, you'll need to update your notebook to use the new files.")
