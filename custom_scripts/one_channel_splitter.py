import os
import sys
import argparse
from skimage import io
from pathlib import Path
from tqdm import tqdm

# Example usage
# python one_channel_splitter.py --indir path/to/indir/ --outdir path/to/outdir/

def get_args_parser():
    parser = argparse.ArgumentParser('channel_splitter', add_help=False)
    parser.add_argument('--indir', type=str, help="""Path to directory containing multi-channel tiffs""")
    parser.add_argument('--outdir', type=str, help="""Path to output directory""")
    return parser
    
def one_channel_splitter(file, indir, outdir):
    # Read the 4-channel image
    path = os.path.join(indir, file)
    img = io.imread(path)

    # Initialize a list to store the paths of the saved images
    saved_images = []

    # Split the image into 4 channels and save each one
    for i in range(img.shape[-1]):
        # Extract the ith channel
        channel_img = img[:, :, i]

        # Construct the filename for the ith channel
        channel_filename = f"{outdir}/{file.split('.')[0]}_channel_{i}.png"
        
        # Save the ith channel as a separate PNG file
        io.imsave(channel_filename, channel_img, check_contrast=False)

        # Append the saved image path to the list
        saved_images.append(channel_filename)

    return saved_images

def process_directory(indir, outdir):
    # List all files in the given directory
    all_files = os.listdir(indir)

    # Process each file
    for file in tqdm(all_files, total=len(all_files), unit="files"):
        # Check if the file name does not contain ']x['
        if ']x[' not in file and '_channel_' not in file: 
            
            saved_image_paths = one_channel_splitter(file, indir, outdir)
            # print(f"Processed {file}")# {saved_image_paths}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('channel_splitter', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    process_directory(args.indir, args.outdir)