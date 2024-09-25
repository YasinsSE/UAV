import os
import glob

# Define the folder path
folder_path = "data/uav-dataset/"

# Get all jpg and txt files
image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
label_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))

# Ensure the number of images matches the number of labels
assert len(image_files) == len(label_files), "Mismatch between image and label files."

# Rename image and label files
for i, (img_file, lbl_file) in enumerate(zip(image_files, label_files), start=1):
    new_img_name = os.path.join(folder_path, f"{i}.jpg")
    new_lbl_name = os.path.join(folder_path, f"{i}.txt")
    
    os.rename(img_file, new_img_name)
    os.rename(lbl_file, new_lbl_name)

    print(f"Renamed: {img_file} to {new_img_name}")
    print(f"Renamed: {lbl_file} to {new_lbl_name}")
