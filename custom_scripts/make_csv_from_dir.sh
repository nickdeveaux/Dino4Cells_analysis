#!/bin/bash

# Directory containing the files
DIR="/home/nick/kaggle_data/test"

# Output CSV file
OUTPUT_CSV="/home/nick/output.csv"

# Write the CSV header
echo "file,ID" > "$OUTPUT_CSV"

# Iterate over files in the directory
for file in "$DIR"/*; do
    # Extract filename without the path
    filename=$(basename "$file")

    # Remove everything after ".png" to get the ID
    id=${filename%.png*}

    # Append the file path with the extracted ID, and the ID to the CSV file
    echo "${file},${id}" >> "$OUTPUT_CSV"
done