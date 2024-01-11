#!/bin/bash

# Directory containing the files
DIR="/home/nick/kaggle_data/train"

# Output CSV file
OUTPUT_CSV="/home/nick/Dino4Cells_analysis/exploratory_configs/kaggle_train_data.csv"

# Write the CSV header
echo "file,ID" > "$OUTPUT_CSV"

declare -A seen


# Iterate over files in the directory
for file in "$DIR"/*; do
    # Extract filename without the path
    filename=$(basename "$file")

    # Remove everything after the first '_' symbol to get the ID
    id=${filename%%_*}

    # Check if this ID has already been processed
    if [[ -z ${seen[$id]} ]]; then
        # Mark this ID as seen
        seen[$id]=1

        # Append the file path with the extracted ID, and the ID to the CSV file
        echo "${file%/*}/${id},${id}" >> "$OUTPUT_CSV"
    fi
done