import pandas as pd
import torch
import numpy as np
from utils.label_dict import protein_to_num_full, protein_to_num_single_cells
from tqdm import tqdm
import argparse
import oyaml as yaml
import os
from pathlib import Path

parser = argparse.ArgumentParser("Get embeddings from model")
parser.add_argument("--config", type=str, default=".", help="path to config file")
parser.add_argument(
    "--use_average",
    action="store_true",
)
parser.add_argument(
    "--cells",
    action="store_true",
)
parser.add_argument(
    "--output_prefix", type=str, default=None, help="path to config file"
)
args = parser.parse_args()
config = yaml.safe_load(open(args.config, "r"))

if config["classification"]["whole_images"]:
    labels = sorted(protein_to_num_full.keys())
else:
    labels = sorted(protein_to_num_single_cells.keys())


if args.use_average:
    feature_dir = config["classification"]["averaged_features_path"]
    train_path = config["classification"]["averaged_train_path"]
    valid_path = config["classification"]["averaged_valid_path"]
else:
    train_path = config["classification"]["train_path"]
    valid_path = config["classification"]["valid_path"]

if args.cells:
    train_path = train_path.replace('.pth','_cells.pth')
    valid_path = valid_path.replace('.pth','_cells.pth')
Path(train_path).parents[0].mkdir(exist_ok=True)
Path(valid_path).parents[0].mkdir(exist_ok=True)

if args.output_prefix is None:
    if args.use_average:
        feature_dir = config["classification"]["averaged_features_path"]
    else:
        feature_dir = config["embedding"]["output_path"]
else:
    feature_dir = args.output_prefix

all_features, all_proteins, all_cell_lines, all_IDs, df = torch.load(feature_dir)
if args.cells:
    train_IDs = torch.load("HPA_FOV_data/cells_train_IDs.pth")
    valid_IDs = torch.load("HPA_FOV_data/cells_valid_IDs.pth")
else:
    train_IDs = torch.load("results/HPA_FOV_classification/unique_filtered_train_IDs.pth")
    valid_IDs = torch.load("results/HPA_FOV_classification/unique_filtered_valid_IDs.pth")

averaged_features = all_features
IDs = np.array(all_IDs)
cell_lines = np.array(all_cell_lines)
if isinstance(all_proteins, list):
    protein_localizations = torch.stack(all_proteins)
if isinstance(all_proteins, torch.Tensor):
    protein_localizations = all_proteins
if isinstance(all_proteins, np.ndarray):
    protein_localizations = torch.Tensor(all_proteins)
sorted_indices = np.argsort(IDs)

averaged_features = averaged_features[sorted_indices, :]
IDs = IDs[sorted_indices]
cell_lines = cell_lines[sorted_indices]
protein_localizations = protein_localizations[sorted_indices]
df = df.sort_values(by="ID").reset_index()

for g, path in zip([train_IDs, valid_IDs], [train_path, valid_path]):
    indices_averaged = np.where(pd.DataFrame(IDs, columns=["ID"]).ID.isin(g))[0]
    torch.save(
        (
            averaged_features[indices_averaged],
            protein_localizations[indices_averaged],
            np.array(cell_lines)[indices_averaged],
            np.array(IDs)[indices_averaged],
        ),
        path,
    )
