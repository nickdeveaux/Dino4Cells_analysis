import csv
from PIL import Image

# Path to the CSV file
csv_file_path = 'external_hpa_fov.csv'

# Function to check if an image is corrupted
def is_image_corrupted(image_path):
    try:
        img = Image.open(image_path)  # Open the image file
        img.verify()  # Verify that it is, in fact, an image
        return False
    except (IOError, SyntaxError) as e:
        print(f"Corrupted image found: {image_path}")
        return True

# Open the CSV file and check each image
with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        file_path = row['file']
        # Check if the image is corrupted
        if is_image_corrupted(file_path):
            print(f"Corrupted file: {file_path}, ID: {row['ID']}")
