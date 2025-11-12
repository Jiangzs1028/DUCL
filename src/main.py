import argparse
from sentence_transformers import SentenceTransformer
from src.data_emb import embeder
from src.wasserstein import data_sort, data_sort_cv
import os
import torch
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--initial_data", type=str)
    parser.add_argument("--source_val", type=str)
    parser.add_argument("--target_val", type=str)
    parser.add_argument("--cache_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--encoder_path", default="sentence-transformers/all-MiniLM-L12-v2", type=str)
    args = parser.parse_args()
    return args

def embedding_load(data_path, cache_path, model_path="sentence-transformers/all-MiniLM-L12-v2"):
    os.makedirs(cache_path, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    result_path = os.path.join(cache_path, f"{file_name}_emb.jsonl")
    if not os.path.exists(result_path):
        model = SentenceTransformer(model_path, device='cuda')
        embeder(model, data_path, result_path)

    dataset = []
    with open(result_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data["embedding"])
    
    return torch.tensor(dataset)

def main():
    args = get_args()
    print("Loading data embeddings")
    data_emb = embedding_load(args.data, args.cache_path, args.encoder_path)
    initial_emb = embedding_load(args.initial_data, args.cache_path)
    source_val_emb = embedding_load(args.source_val, args.cache_path)
    if args.target_val is not None:
        target_val_emb = embedding_load(args.target_val, args.cache_path)

    file_name = os.path.splitext(os.path.basename(args.data))[0]
    grad_path = os.path.join(args.cache_path, f"{file_name}_grad.jsonl")
    print("Starting Wasserstein distance calculation")
    
    if args.target_val is not None:
        source_gradient, target_gradient = data_sort(data_emb, initial_emb, source_val_emb, target_val_emb, grad_path, device='cuda')
    else:
        source_gradient, target_gradient = data_sort_cv(data_emb, initial_emb, source_val_emb, grad_path, device='cuda')

    dataset = []
    with open(args.data, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            data["difficulty"] = float(source_gradient[i])
            data["utility"] = float(target_gradient[i])
            data["DUE"] = float(source_gradient[i])/float(target_gradient[i])
            dataset.append(data)

    with open(args.result_path, 'w') as file:
        for i in range(len(dataset)):
            json_record = json.dumps(dataset[i], ensure_ascii=False)
            file.write(json_record + '\n')

