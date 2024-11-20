import os
import json
import shutil

# Paths (replace with your actual paths)
json_path = 'path/to/your/json_file.json'  # Replace with the path to your JSON file
images_dir = 'path/to/your/images_directory'  # Replace with the path to your images directory
output_dir = 'path/to/output/folder'  # Replace with the path where you want the new folders

# Create output directories if they don't exist
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Move images based on their subset
for subset, images in data.items():
    if subset == "train":
        target_dir = train_dir
    elif subset == "val":
        target_dir = val_dir
    elif subset == "test":
        target_dir = test_dir
    else:
        continue  # Ignore unknown subsets

    # Move each image to the respective folder
    for image_name in images:
        src_path = os.path.join(images_dir, image_name)
        dest_path = os.path.join(target_dir, image_name)
        
        # Check if the image file exists before moving
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
            print(f"Moved {image_name} to {target_dir}")
        else:
            print(f"Image {image_name} not found in {images_dir}.")

print("Images have been sorted into train, val, and test folders.")