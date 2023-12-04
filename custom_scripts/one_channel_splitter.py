from skimage import io

def one_channel_splitter(path):
    # Read the 4-channel image
    img = io.imread(path)

    # Initialize a list to store the paths of the saved images
    saved_images = []

    # Split the image into 4 channels and save each one
    for i in range(img.shape[-1]):
        # Extract the ith channel
        channel_img = img[:, :, i]

        # Construct the filename for the ith channel
        channel_filename = f"{path.split('.')[0]}_channel_{i}.png"

        # Save the ith channel as a separate PNG file
        io.imsave(channel_filename, channel_img)

        # Append the saved image path to the list
        saved_images.append(channel_filename)

    return saved_images

# Example usage
path = "/home/nick/custom_scripts/path/to/output_dir/74391_1503_C3_1.png"
saved_image_paths = one_channel_splitter(path)