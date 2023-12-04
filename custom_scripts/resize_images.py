import os
from skimage import io, transform
import numpy as np

def resize_and_convert_images(directory):
    for filename in os.listdir(directory):
        if ']x[' in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Construct the full file path
            file_path = os.path.join(directory, filename)

            try:
                # Read the image using skimage
                img = io.imread(file_path)

                # Resize the image to 512x512
                img_resized = transform.resize(img, (512, 512), anti_aliasing=True)

                # Ensure the image is in the right format (e.g., uint8)
                if img_resized.max() <= 1.0:
                    img_resized = (img_resized * 255).astype(np.uint8)

                # Construct new filename (replace original extension with .tiff)
                new_name = filename.split('.')[0] + ".tiff"

                # Save the resized image in TIFF format using skimage
                io.imsave(os.path.join(directory, new_name), img_resized)

            except Exception as e:
                # Print the error message and the file name, then continue with the next image
                print(f"Error processing {filename}: {e}")
                continue


directory = "/home/nick/custom_scripts/path/to/output_dir"
resize_and_convert_images(directory)
