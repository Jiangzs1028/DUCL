import torch
from geomloss import SamplesLoss
from sklearn.model_selection import KFold
import numpy as np
import json
import time
import os

def offset(values):
    """Return the offset to make all values positive."""
    min_val = np.min(values)
    return abs(min_val) + 1e-8 if min_val < 0 else 0.0


def data_value(data_emb, initial_emb, source_val_emb, target_val_emb, source_weight, device, epsilon=0.2):
    data_emb       = data_emb.to(device)
    initial_emb    = initial_emb.to(device)
    source_val_emb = source_val_emb.to(device)
    target_val_emb = target_val_emb.to(device)

    combined_data = torch.cat((initial_emb, data_emb), dim=0)

    ot_loss = SamplesLoss(
        loss='sinkhorn',
        p=2,
        blur=0.1,
        debias=False,
        verbose=False,
        potentials=True,
    )

    train_weights = torch.zeros(len(combined_data), device=device)

    if source_weight is not None:
        sw = source_weight.to(device)
        assert torch.all(sw >= 0), "source_weight must be non-negative"

        train_weights[:len(initial_emb)] = (1 - epsilon) * sw
        source_val_weights = sw
    else:
        train_weights[:len(initial_emb)] = (1 - epsilon) / len(initial_emb)
        source_val_weights = torch.full(
            (len(source_val_emb),), 1.0 / len(source_val_emb), device=device
        )

    train_weights[len(initial_emb):] = epsilon / len(data_emb)

    target_val_weights = torch.full(
        (len(target_val_emb),), 1.0 / len(target_val_emb), device=device
    )
    start_time = time.time()
    with torch.no_grad():
        F_source, _ = ot_loss(train_weights, combined_data, source_val_weights, source_val_emb)
        F_target, _ = ot_loss(train_weights, combined_data, target_val_weights, target_val_emb)

    ot_time = time.time() - start_time
    print(f"Time taken for Wasserstein distance computation: {ot_time}")
    f_difficulty = np.array(F_source.cpu().squeeze())
    f_utility = -np.array(F_target.cpu().squeeze())

    difficulty = f_difficulty[len(initial_emb):]
    utility = f_utility[len(initial_emb):]

    # Ensure all metrics are positive, as only the relative magnitude between samples matters
    difficulty = difficulty + offset(difficulty)
    utility = utility + offset(utility) + 1e-3

    return difficulty, utility


