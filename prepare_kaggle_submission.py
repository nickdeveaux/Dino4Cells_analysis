import os
import time
import sys
import math
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from skimage import io

import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CyclicLR
from sklearn.metrics import f1_score

# custom code
from utils.label_dict import protein_to_num_full, protein_to_num_single_cells
from utils.classification_utils import get_classifier
from utils.utils import init_distributed_mode
from utils.utils import is_main_process
from utils import vision_transformer as vits
import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib
from utils.file_dataset import (
    ImageFileList,
    AutoBalancedFileList,
    AutoBalancedPrecomputedFeatures,
    default_loader,
    pandas_reader,
    pandas_reader_binary_labels,
    pandas_reader_no_labels,
    scKaggle_df_reader,
)


def filter_cell(mask, threshold):
    left_limit = np.where(mask.sum(axis=0) > 0)[0][0]
    right_limit = np.where(mask.sum(axis=0) > 0)[0][-1]
    upper_limit = np.where(mask.sum(axis=1) > 0)[0][0]
    lower_limit = np.where(mask.sum(axis=1) > 0)[0][-1]
    if left_limit < threshold:
        return True
    if upper_limit < threshold:
        return True
    if (mask.shape[0] - right_limit) < threshold:
        return True
    if (mask.shape[1] - lower_limit) < threshold:
        return True
    return False


# Separate cell ID from whole image ID
def disentangle_ID(ID):
    return "_".join(ID.split("_")[:-1]), int(ID.split("_")[-1])


def run(
    config,
    **kwargs,
):
    # Populate arg namespace with config parameters
    args = argparse.Namespace()
    if config.endswith("yaml"):
        config = yaml.safe_load(open(config, "r"))
    else:
        config = json.load(open(config))
        for k in config["command_lind_args"].keys():
            config["classification"][k] = config["command_lind_args"][k]
    for k in config["classification"].keys():
        setattr(args, k, config["classification"][k])

    # Override config file with command line arguments
    for k in kwargs.keys():
        if k in args.__dict__.keys():
            print(
                f"Command line arguments overridden config file {k} (was {args.__dict__[k]}, is now {kwargs[k]})"
            )
        setattr(args, k, kwargs[k])

    # if there are multiple gpus, distribute model
    if args.parallel_training:
        init_distributed_mode(args)

    # Make sure bool arguments are treated as bool
    if type(args.whole_images) == str:
        args.whole_images = args.whole_images == "True"
    if "skip" in args and type(args.skip) == str:
        args.skip = args.skip == "True"

    print(f"Output_dir: {args.output_dir}, output prefix: {args.output_prefix}")
    save_dir = f"{args.output_dir}/{args.output_prefix}/"
    if Path(save_dir).exists() and args.overwrite == False:
        print(
            f"\n\nError: Folder {args.output_dir}/{args.output_prefix} exists; Please change experiment name to prevent loss of information\n\n"
        )
        quit()
    # wait for all processes to check for existing folder
    time.sleep(1)

    if is_main_process():
        Path(save_dir).mkdir(exist_ok=True)
        log_params(
            config,
            kwargs,
            log_path=f"{save_dir}/experiment_params.json",
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    class simple_clf(nn.Module):
        def __init__(self, n_features, n_classes, p=0, with_sigmoid=False):
            super().__init__()
            self.p = p
            self.with_sigmoid = with_sigmoid
            self.clf = nn.Sequential(
                nn.Linear(n_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(256, n_classes),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.clf(x)
            if self.with_sigmoid:
                x = self.sigmoid(x)
            return x

    # setup classifier head
    if args.whole_images:
        args.test_path = args.averaged_test_path
    embed_dim = torch.load(args.test_path)[0].shape[1]
    classifier = get_classifier(args, embed_dim)
    # args.classifier_state_dict = args.classifier_state_dict.replace(
    #     "protein", "protein_whole" if args.whole_images else "protein"
    # )
    # print(args.classifier_state_dict)
    msg = classifier.load_state_dict(
        torch.load(
            args.classifier_state_dict,
            map_location="cpu",
        )
    )
    print(
        "Pretrained weights for classifier found at {} and loaded with msg: {}".format(
            args.classifier_state_dict, msg
        )
    )
    classifier.to(device)

    if args.whole_images:
        target_labels = sorted(list(protein_to_num_full.keys()))
    else:
        target_labels = sorted(list(protein_to_num_single_cells.keys()))

    for p in classifier.parameters():
        p.requires_grad = False
    classifier = classifier.eval()

    # Make sure models predict correct column order
    # (needed for older models)
    if args.competition_type in [
        "single_cells",
        "single_cells_with_aggregated_prediction",
        "single_cells_with_test_time_augmentations",
    ]:
        mapping_dict = protein_to_num_single_cells
    else:
        mapping_dict = protein_to_num_full

    features, IDs, impaths = torch.load(args.test_path)
    if Path(f"{args.output_dir}/{args.output_prefix}/scaler.pth").exists():
        print("scaling features")
        scaler = torch.load(f"{args.output_dir}/{args.output_prefix}/scaler.pth")
        features = scaler.transform(features.numpy())
    features = torch.Tensor(features)
    target_labels = sorted(list(mapping_dict.keys()))
    mapping = np.arange(args.num_classes)
    for ind, p in enumerate(list(mapping_dict.keys())):
        mapping[ind] = target_labels.index(p)
    predictions = []

    # Generate predictions from features
    for f in features:
        student_output = classifier(f.unsqueeze(0).to(device))
        predictions.append(student_output.cpu().detach().numpy())
    if args.competition_type == "whole_images":
        predictions = np.concatenate(predictions, axis=0)
        predictions = (
            torch.sigmoid(torch.Tensor(predictions)).round().squeeze(1)[:, mapping]
        )
        submissions = []
        for i in predictions:
            if len(i) == 0:
                submissions.append("")
            else:
                submissions.append(" ".join([str(v) for v in np.where(i)[0]]))

        submission = pd.DataFrame(zip(IDs, submissions), columns=["Id", "Predicted"])
        submission = submission.sort_values(by="Id")

    submission_file_path = (
        f"{args.output_dir}/{args.output_prefix}/submission_{args.competition_type}.csv"
    )
    submission.to_csv(
        submission_file_path,
        index=False,
    )
    print(f"Successfully wrote kaggle submission file to {submission_file_path}")


# Log experiment parameters
def log_params(config, args, log_path):
    config["command_lind_args"] = args
    with open(log_path, "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINO")
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--edge_threshold", default=None, type=int)
    # parse unrecognized parameters
    args, unknown = parser.parse_known_args()
    keys = unknown[0::2]
    values = unknown[1::2]
    keys += ["parallel_training"]
    values += [True if "WORLD_SIZE" in os.environ.keys() else False]
    for k, v in zip(keys, values):
        setattr(args, k.replace("--", ""), v)
    run(
        args.config,
        **{k: args.__dict__[k] for k in args.__dict__.keys() if "config" not in k},
    )
