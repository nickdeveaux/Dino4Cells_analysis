import hashlib
import os

def file_hash(filename):
    """Compute hash of a file."""
    hasher = hashlib.md5()
    with open(filename, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def find_duplicates(directory):
    """Find and count duplicate files in a directory."""
    hashes = {}
    duplicates = {}

    for filename in os.listdir(directory):
        if filename.endswith('.pt'):  # Only process .pt files
            path = os.path.join(directory, filename)
            filehash = file_hash(path)

            if filehash in hashes:
                if filehash in duplicates:
                    duplicates[filehash].append(filename)
                else:
                    duplicates[filehash] = [hashes[filehash], filename]
            else:
                hashes[filehash] = filename

    return duplicates

# Usage example
directory = '/home/nick/tensor_cache'
duplicates = find_duplicates(directory)

for hash, filenames in duplicates.items():
    print(f"Duplicate files: {filenames}")