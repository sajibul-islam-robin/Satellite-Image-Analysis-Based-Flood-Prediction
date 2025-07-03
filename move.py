import os
import shutil

# Define paths
masks_dir = r"C:\Users\sajib\OneDrive\Desktop\Flood_Prediction\data\test\masks\train_masks_flood"
images_dir = r"C:\Users\sajib\OneDrive\Desktop\Flood_Prediction\data\test\images"
target_dir = os.path.join(images_dir, "train_images_flood")

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Get mask filenames without extensions
mask_names = {os.path.splitext(f)[0] for f in os.listdir(masks_dir)}

# Iterate over image files and move matching ones
for file in os.listdir(images_dir):
    file_path = os.path.join(images_dir, file)
    if os.path.isfile(file_path):
        base_name = os.path.splitext(file)[0]
        if base_name in mask_names:
            shutil.move(file_path, os.path.join(target_dir, file))
            print(f"Moved: {file}")
