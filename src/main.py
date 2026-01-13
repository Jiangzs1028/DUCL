import argparse
from sentence_transformers import SentenceTransformer
from src.data_emb import embeder
from src.wasserstein import data_value
import os
import torch
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Training data to be evaluated and ordered.")
    parser.add_argument("--source_val", type=str, help="Validation set from the source data distribution.")
    parser.add_argument("--target_val", type=str, help="Validation set from the target data distribution.")
    parser.add_argument("--cache_path", type=str, help="Path to the temporary cache directory.")
    parser.add_argument("--result_path", type=str, help="Path to save the final dataset annotated with DUE scores.")
    parser.add_argument("--encoder_path", default="sentence-transformers/all-MiniLM-L12-v2", type=str, help="Path or name of the embedding model.")
    parser.add_argument("--batch_size", default=200, type=int, help="Batch size used for embedding computation.")
    parser.add_argument("--chunk_num", default=1, type=int, help="Number of chunks to split each input text into when it exceeds the encoder context limit.")
    parser.add_argument("--is_approx", default=False, type=int, help="Approximate the source distribution.")    
    args = parser.parse_args()
    return args

def embedding_load(data_path, args):
    os.makedirs(args.cache_path, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(data_path))[0]
    result_path = os.path.join(args.cache_path, f"{file_name}_emb.jsonl")

    model = SentenceTransformer(args.encoder_path, device='cuda')
    embeder(model, data_path, result_path, batch_size=args.batch_size, chunk_num=args.chunk_num)

    dataset = []
    with open(result_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data["embedding"])
    
    return torch.tensor(dataset)

def get_approx_weight(data_path, temperature=1.0):
    loss_list = []
    with open(data_path, "r") as file:
        for line in file:
            data = json.loads(line)
            loss_list.append(data["loss"])

    loss = torch.tensor(loss_list, dtype=torch.float32)

    logits = -loss / temperature
    logits = logits - logits.max()
    weights = torch.softmax(logits, dim=0)

    return weights



def main():
    args = get_args()
    print("Loading data embeddings")
    data_emb = embedding_load(args.data, args)
    source_val_emb = embedding_load(args.source_val, args)
    initial_emb = source_val_emb
    target_val_emb = embedding_load(args.target_val, args)

    print("Starting Wasserstein distance calculation")

    source_weight = get_approx_weight(args.source_val) if args.is_approx else None


    difficulty, utility = data_value(data_emb, initial_emb, source_val_emb, target_val_emb, source_weight, device='cuda')

    print("Completed Wasserstein distance calculation.")

    print("Dataset Saving...")
    dataset = []
    with open(args.data, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            data["difficulty"] = float(difficulty[i])
            data["utility"] = float(utility[i])
            data["DUE"] = float(difficulty[i])/float(utility[i])
            dataset.append(data)

    with open(args.result_path, 'w') as file:
        for i in range(len(dataset)):
            json_record = json.dumps(dataset[i], ensure_ascii=False)
            file.write(json_record + '\n')
    print("Done!")
