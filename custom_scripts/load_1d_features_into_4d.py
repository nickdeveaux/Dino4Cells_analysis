import torch

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

def save_new_data(new_tensors, new_ids, output_file):
    all_features = torch.stack(new_tensors)
    result = [all_features, new_ids]
    torch.save(result, output_file)


# Load the data
file_path = 'results_01_09_from_features_1channel_dino/results_01_09_from_features_1channel_dino.pth'
tensors, ids = load_data(file_path)

# Process the data
new_tensors, new_ids = process_data(tensors, ids)

# Save the new data
output_file = 'processed_data.pth'
save_new_data(new_tensors, new_ids, output_file)