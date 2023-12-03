directory = "/home/nick/custom_scripts/path/to/output_dir"
import os
from PIL import Image

def resize_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and ']x[' not in filename:
            # Construct the full file path
            file_path = os.path.join(directory, filename)

            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Get original size
                    original_size = img.size

                    # Skip resizing if image is already 512x512
                    if original_size == (512, 512):
                        continue

                    # Rename the original image
                    new_name = f"{filename.split('.')[0]}_{[0]}x{[1]}.{filename.split('.')[-1]}"
                    os.rename(file_path, os.path.join(directory, new_name))

                    # Resize the image
                    img = img.resize((512, 512), Image.ANTIALIAS)

                    # Save the resized image
                    img.save(os.path.join(directory, filename))  # This will overwrite the original image

            except OSError as e:
                # Print the error message and the file name, then continue with the next image
                print(f"Error processing {filename}: {e}")
                continue

resize_images(directory) 