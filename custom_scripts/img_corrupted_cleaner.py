# Given script is modified to write non-corrupted entries into a new CSV file

import csv
from PIL import Image

# Path to the original CSV file and the new clean CSV file
original_csv_path = 'external_hpa_fov.csv'
clean_csv_path = 'external_hpa_fov_cleaned.csv'

# Function to check if an image is corrupted
def is_image_corrupted(image_path):
    try:
        img = Image.open(image_path)  # Open the image file
        img.verify()  # Verify that it is, in fact, an image
        return False
    except (IOError, SyntaxError) as e:
        return True

# Open the original CSV file and create a new cleaned CSV file
with open(original_csv_path, mode='r') as infile, open(clean_csv_path, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        file_path = row['file']
        # Check if the image is corrupted and write to the clean CSV if not
        if not is_image_corrupted(file_path):
            writer.writerow(row)
        else:
            print(f"Corrupted file: {file_path}, ID: {row['ID']}")

clean_csv_path  # Output the path of the cleaned CSV file
