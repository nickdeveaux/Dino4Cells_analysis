import os
from PIL import Image

def resize_images(directory):
    for filename in os.listdir(directory):
        if ']x[' in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Construct the full file path
            file_path = os.path.join(directory, filename)

            try:
                # Open the image
                with Image.open(file_path) as img:

                    img = img.resize((512, 512), Image.ANTIALIAS)

                        # Remove single quotes, '_[0]x[1]', and file extension from the original filename
                    new_name = filename.replace("'", "").split('_[0]x[1]')[0] + ".png"

                        # Save the resized image with the new filename
                    img.save(os.path.join(directory, new_name))

            except OSError as e:
                # Print the error message and the file name, then continue with the next image
                print(f"Error processing {filename}: {e}")
                continue

directory = "/home/nick/custom_scripts/path/to/output_dir"
resize_images(directory)
