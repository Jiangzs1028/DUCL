import torch
from geomloss import SamplesLoss
from sklearn.model_selection import KFold
import numpy as np
import json
import time
import os

def mix(values):
    """Return the offset to make all values positive."""
    min_val = np.min(values)
    return abs(min_val) + 1e-8 if min_val < 0 else 0.0


def data_sort(data_emb, initial_emb, source_val_emb, target_val_emb, result_path, device):
    
    combined_data = torch.cat((initial_emb, data_emb), dim=0)

    # Initialize SamplesLoss
    ot_loss = SamplesLoss(
        loss='sinkhorn',    # Use Sinkhorn loss
        p=2,                # 2-Wasserstein distance
        blur=0.1,           # Sinkhorn regularization parameter
        # scaling=0.5,      # Scaling factor for the multiscale scheme
        # truncate=5,       # Kernel truncation parameter
        debias=False,       # Use unbiased estimation
        verbose=False,      # Do not display verbose output
        potentials=True,
        # backend='tensorized'
    )
        # test_data = combined_tensor[test_idx].to(device)
    
    # Create weights
    train_weights = torch.zeros(len(combined_data), device=device)

    train_weights[:len(initial_emb)] = 0.8 / len(initial_emb)
    train_weights[len(initial_emb):] = 0.2 / len(data_emb)

    source_val_weights = torch.full((len(source_val_emb),), 1 / source_val_emb, device=device)
    target_val_weights = torch.full((len(target_val_emb),), 1 / target_val_emb, device=device)

    with torch.no_grad():
        # Compute OT distance with weights
        start_time = time.time()
        F_source, _ = ot_loss(train_weights, combined_data, source_val_weights, source_val_emb)
        F_target, _ = ot_loss(train_weights, combined_data, target_val_weights, target_val_emb)

    ot_time = time.time() - start_time
    print(f"Time taken for Wasserstein distance computation: {ot_time}")
    f_source = np.array(F_source.cpu().squeeze())
    f_target = -np.array(F_target.cpu().squeeze())

    source_gradient = f_source - sum(f_source[:len(initial_emb)]) / len(initial_emb)
    target_gradient = f_target - sum(f_target[:len(initial_emb)]) / len(initial_emb)

    source_gradient = source_gradient[len(initial_emb):]
    target_gradient = target_gradient[len(initial_emb):]

    # Ensure all gradients are positive
    offset_s = mix(source_gradient)
    offset_t = mix(target_gradient)
    source_gradient = source_gradient + offset_s
    target_gradient = target_gradient + offset_t



    with open(result_path, 'w') as file:
        for i in range(len(source_gradient)):
            jsonl = {"source_grad": source_gradient[i], "target_grad": target_gradient[i]}
            json_record = json.dumps(jsonl, ensure_ascii=False)
            file.write(json_record + '\n')

    return source_gradient, target_gradient



def data_sort_cv(data_emb, initial_emb, source_val_emb, result_path, device):
    # If the processed result already exists, load and return it directly
    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists, loading results directly.")
        with open(result_path, 'r', encoding='utf-8') as file:
            source_grad_mean = []
            target_grad_mean = []
            for line in file:
                data = json.loads(line)
                source_grad_mean.append(data["source_grad"])
                target_grad_mean.append(data["target_grad"])
            return source_grad_mean, target_grad_mean
        
    n = len(data_emb)
    kf = KFold(n_splits=4, shuffle=False)
    
    source_val_emb = source_val_emb.to(device)
    
    # Initialize accumulators and counters
    source_grad_accumulator = torch.zeros(n, device=device)
    target_grad_accumulator = torch.zeros(n, device=device)
    count = torch.zeros(n, device=device)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_emb)):
        test_idx = test_idx[:min(5000, len(test_idx))]

        # Split dataset
        data_emb_part = data_emb[train_idx]
        target_val_emb_part = data_emb[test_idx].to(device)
        
        # Concatenate combined data
        combined_data = torch.cat((initial_emb, data_emb_part), dim=0).to(device)
        
        # Initialize OT loss function
        ot_loss = SamplesLoss(
            loss='sinkhorn',
            p=2,
            blur=0.1,
            debias=False,
            verbose=False,
            potentials=True
        )
        
        # Create weight vectors
        train_weights = torch.zeros(len(combined_data), device=device)
        train_weights[:len(initial_emb)] = 0.8 / len(initial_emb)
        train_weights[len(initial_emb):] = 0.2 / len(data_emb_part)
        
        source_val_weights = torch.full(
            (len(source_val_emb),), 
            1.0 / len(source_val_emb), 
            device=device
        )
        target_val_weights = torch.full(
            (len(target_val_emb_part),), 
            1.0 / len(target_val_emb_part), 
            device=device
        )
        
        # Compute OT distance
        with torch.no_grad():
            start_time = time.time()
            F_source, _ = ot_loss(train_weights, combined_data, source_val_weights, source_val_emb)
            F_target, _ = ot_loss(train_weights, combined_data, target_val_weights, target_val_emb_part)
        
        ot_time = time.time() - start_time
        print(f"Fold {fold+1} Wasserstein distance computation time: {ot_time}")
        
        # Convert to numpy arrays
        f_source = F_source.cpu().numpy().squeeze()
        f_target = -F_target.cpu().numpy().squeeze()
        
        # Compute gradients (keep only the data_emb part)
        source_gradient = f_source[len(initial_emb):] - np.mean(f_source[:len(initial_emb)])
        target_gradient = f_target[len(initial_emb):] - np.mean(f_target[:len(initial_emb)])
        
        # Accumulate results to the original index positions
        source_grad_accumulator[train_idx] += torch.from_numpy(source_gradient).to(device)
        target_grad_accumulator[train_idx] += torch.from_numpy(target_gradient).to(device)
        count[train_idx] += 1
    
    # Compute averaged results
    source_grad_mean = (source_grad_accumulator / 3).cpu().numpy()
    target_grad_mean = (target_grad_accumulator / 3).cpu().numpy()

    # Ensure all averaged gradients are positive
    offset_s = mix(source_grad_mean)
    offset_t = mix(target_grad_mean)
    source_grad_mean = source_grad_mean + offset_s
    target_grad_mean = target_grad_mean + offset_t
    
    # Write results to file
    with open(result_path, 'w') as file:
        for s_grad, t_grad in zip(source_grad_mean, target_grad_mean):
            jsonl = {"source_grad": float(s_grad), "target_grad": float(t_grad)}
            file.write(json.dumps(jsonl) + '\n')
    
    return source_grad_mean, target_grad_mean
