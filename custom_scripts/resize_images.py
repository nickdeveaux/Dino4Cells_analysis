import os
from skimage import io, transform
import numpy as np

def resize_images(directory, convert=False):
    for filename in os.listdir(directory):
        if ']x[' in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Construct the full file path
            file_path = os.path.join(directory, filename)

            # Construct new filename (replace original extension with .tiff)
            # Remove single quotes, '_[0]x[1]', and file extension from the original filename
            new_name = filename.replace("'", "").split('_[0]x[1]')[0]
            new_name = new_name + ".tiff" if convert else new_name + ".png"
            new_file_path = os.path.join(directory, new_name)

            # Check if new_name already exists
            if os.path.exists(new_file_path):
                print(f"File already exists: {new_name}, skipping...")
                continue

            try:
                # Read the image using skimage
                img = io.imread(file_path)

                # Resize the image to 512x512
                img_resized = transform.resize(img, (512, 512), anti_aliasing=True)

                # Ensure the image is in the right format (e.g., uint8)
                if img_resized.max() <= 1.0:
                    img_resized = (img_resized * 255).astype(np.uint8)

                # Save the resized image in TIFF or PNG format using skimage
                io.imsave(os.path.join(directory, new_name), img_resized)

            except Exception as e:
                # Print the error message and the file name, then continue with the next image
                print(f"Error processing {filename}: {e}")
                continue


directory = "/home/nick/custom_scripts/path/to/output_dir"
resize_images(directory)
