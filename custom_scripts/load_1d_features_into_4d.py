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


# Load the data
file_path = 'results_01_09_from_features_1channel_dino/results_01_09_from_features_1channel_dino.pth'
tensors, ids = load_data(file_path)

# Process the data
new_tensors, new_ids = process_data(tensors, ids)

cell_and_protein_path = '/home/nick/data/HPA_FOV_data/4channel_12_18_DINO_features_for_HPA_FOV.pth'
reordered_tensor_list_1, reordered_tensor_list_2 = load_and_reorder_data(cell_and_protein_path, new_ids)

# Save the new data
output_file = 'processed_data.pth'
save_new_data(new_tensors, reordered_tensor_list_1, reordered_tensor_list_2, new_ids, output_file)