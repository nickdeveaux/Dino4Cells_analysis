import torch


def load_and_reorder_data(file_path, new_ids):
    # Load the data
    data = torch.load(file_path)
    features, tensor_list_1, tensor_list_2, ids = data

    # Create a mapping from ID to index
    id_to_index = {id_str: i for i, id_str in enumerate(ids)}

    # Reorder tensor_list_1 and tensor_list_2 according to new_ids
    reordered_tensor_list_1 = [tensor_list_1[id_to_index[id_str]] for id_str in new_ids if id_str in id_to_index]
    reordered_tensor_list_2 = [tensor_list_2[id_to_index[id_str]] for id_str in new_ids if id_str in id_to_index]

    return reordered_tensor_list_1, reordered_tensor_list_2


def load_data(file_path):
    data = torch.load(file_path)
    tensors, ids = data
    return tensors, ids


def process_kaggle_color_data(tensors, ids):
    grouped_tensors = {}
    for tensor, id_str in zip(tensors, ids):
        # Extract the base ID without the color channel
        base_id = id_str.rsplit('_', 1)[0]

        # Initialize a dictionary for this ID if not already present
        if base_id not in grouped_tensors:
            grouped_tensors[base_id] = {'blue': None, 'green': None, 'red': None, 'yellow': None}

        # Determine the color channel of the current tensor
        channel = id_str.split('_')[-1]

        # Assign the tensor to the correct channel
        if channel in grouped_tensors[base_id]:
            grouped_tensors[base_id][channel] = tensor

    new_tensors = []
    new_ids = []
    for base_id, channels in grouped_tensors.items():
        # Check if all four channels have tensors assigned
        if all(channels[ch] is not None for ch in ['blue', 'green', 'red', 'yellow']):
            # Concatenate tensors in the order of 'blue', 'green', 'red', 'yellow'
            concatenated_tensor = torch.cat([channels[ch] for ch in ['blue', 'green', 'red', 'yellow']], dim=0)
            new_tensors.append(concatenated_tensor)
            new_ids.append(base_id)

    return new_tensors, new_ids


def process_data(tensors, ids):
    grouped_tensors = {}
    for tensor, id_str in zip(tensors, ids):
        base_id = id_str.rsplit('_channel', 1)[0]  # Remove the channel part
        if base_id not in grouped_tensors:
            grouped_tensors[base_id] = []
        grouped_tensors[base_id].append(tensor)

    new_tensors = []
    new_ids = []
    for base_id, tensor_group in grouped_tensors.items():
        if len(tensor_group) == 4:  # Ensure we have 4 channels
            concatenated_tensor = torch.cat(tensor_group, dim=0)
            new_tensors.append(concatenated_tensor)
            new_ids.append(base_id)

    return new_tensors, new_ids


def save_new_data(new_tensors, tensor_list_1, tensor_list_2, new_ids, output_file):
    all_features = torch.stack(new_tensors)
    result = [all_features, tensor_list_1, tensor_list_2, new_ids]
    torch.save(result, output_file)


def save_data_with_just_features_and_ids(new_tensors, new_ids, output_file):
    all_features = torch.stack(new_tensors)
    result = [all_features, new_ids]
    torch.save(result, output_file)



def load_and_reorder_data(file_path, new_ids):
    # Load the data
    data = torch.load(file_path)
    features, tensor_list_1, tensor_list_2, ids = data

    # Create a mapping from ID to index
    id_to_index = {id_str: i for i, id_str in enumerate(ids)}

    # Reorder tensor_list_1 and tensor_list_2 according to new_ids
    # Also collect the IDs that are actually present in the data
    reordered_tensor_list_1 = []
    reordered_tensor_list_2 = []
    present_ids = []
    for id_str in new_ids:
        if id_str in id_to_index:
            index = id_to_index[id_str]
            reordered_tensor_list_1.append(tensor_list_1[index])
            reordered_tensor_list_2.append(tensor_list_2[index])
            present_ids.append(id_str)

    return reordered_tensor_list_1, reordered_tensor_list_2, present_ids

# Load the data
file_path = 'results_01_09_from_features_1channel_dino/results_01_09_from_features_1channel_dino.pth'
tensors, ids = load_data(file_path)

# Process the data
new_tensors, new_ids = process_data(tensors, ids)

cell_and_protein_path = '/home/nick/data/HPA_FOV_data/4channel_12_18_DINO_features_for_HPA_FOV.pth'
reordered_tensor_list_1, reordered_tensor_list_2, present_ids = load_and_reorder_data(cell_and_protein_path, new_ids)

# Filter new_tensors and new_ids based on present_ids
filtered_new_tensors = [tensor for tensor, id_str in zip(new_tensors, new_ids) if id_str in present_ids]
filtered_new_ids = [id_str for id_str in new_ids if id_str in present_ids]

# Check if all the lists have the same length
if not (len(filtered_new_tensors) == len(reordered_tensor_list_1) == len(reordered_tensor_list_2) == len(filtered_new_ids)):
    print(f"Lengths are not the same:")
    print(f"Length of filtered_new_tensors: {len(filtered_new_tensors)}")
    print(f"Length of reordered_tensor_list_1: {len(reordered_tensor_list_1)}")
    print(f"Length of reordered_tensor_list_2: {len(reordered_tensor_list_2)}")
    print(f"Length of filtered_new_ids: {len(filtered_new_ids)}")
    raise ValueError("The lengths of the vectors to be saved are not the same. Aborting save operation.")
else:
    # Save the new data
    output_file = 'processed_data.pth'
    save_new_data(filtered_new_tensors, reordered_tensor_list_1, reordered_tensor_list_2, filtered_new_ids, output_file)
    data_length = len(filtered_new_tensors)  # or len(filtered_new_ids), as they should be the same
    print(f"Data saved successfully to {output_file}. Number of entries: {data_length}")