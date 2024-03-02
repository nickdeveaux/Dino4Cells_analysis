import umap.umap_ as umap
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.label_dict import protein_to_num_full
from matplotlib import cm
cmap = cm.nipy_spectral
from utils.classification_utils import get_classifier, FocalBCELoss, threshold_output, write_to_tensorboard, get_scheduler, get_optimizer
from sklearn.metrics import f1_score
reducer = umap.UMAP(random_state=42)



def plot_UMAP(df, labels, embedding, title, color_indices):
    mat, labels = get_col_matrix(df, labels)
    plt.figure(figsize=(10, 10), facecolor='white', dpi=300)
    plt.axis(False)
    plt.scatter(embedding[:, 0],
                embedding[:, 1],
                s=0.01,
                label='All data',
                color='grey'
               )
    for i, ind in enumerate(np.argsort(mat.sum(axis=0))[::-1]):
        indices = np.where((mat[:, ind] == 1) & (mat.sum(axis=1) == 1))[0]
        plt.scatter(embedding[indices, 0],
                    embedding[indices, 1],
                    s=0.01,
                    label=labels[ind],
                    color=cmap(color_indices[ind] / mat.shape[1])
                   )

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(title, fontsize=15)
    lgnd = plt.legend(bbox_to_anchor=(1, 1), frameon=False)
    for h in lgnd.legendHandles:
        h._sizes = [30]

    # Display the plot
    plt.show()

fts = torch.load("/mnt/vast/hpc/LDB/LDB_data/microscopy/HPA/whole_images/DINO_features_for_HPA_FOV.pth")
train_idx = torch.load("/mnt/vast/hpc/LDB/LDB_data/microscopy/HPA/whole_images/fovHPA_train_IDs_proteinloc.pth")
valid_idx = torch.load("/mnt/vast/hpc/LDB/LDB_data/microscopy/HPA/whole_images/fovHPA_valid_IDs_proteinloc.pth")
train_fts, valid_fts = [], []
for n in range(4):
    x = fts[n]
    if isinstance(x, list): x = np.array(x)
    train_fts.append(x[~fts_ids.isin(valid_idx)])
    valid_fts.append(x[fts_ids.isin(valid_idx)])
import pdb; pdb.set_trace()
embedding = reducer.fit_transform(train_fts[0][0:150])
