from multiprocessing import Process, Queue, Pool
import queue
import threading
from typing import List, Dict
import time
from sentence_transformers import SentenceTransformer
import json
from torch.utils.data import Dataset, DataLoader
import tqdm
import os
import torch

NUM_TOKENIZER_WORKERS = 16  # Set the number of tokenizer worker processes
chunk_num = 8


class SentenceDataset(Dataset):
    def __init__(self, data_path):
        self.sentences = []
        num = 0
        with open(data_path, 'r') as file:
            for line in file:
                num += 1
                jsonline = json.loads(line)
                if 'text' in jsonline:
                    item = {'index': num, 'text': jsonline["text"]}
                else:
                    item = {'index': num, 'text': jsonline['problem'] + jsonline['deepseek_reasoning'] + jsonline['deepseek_solution']}
                self.sentences.append(item)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
    

class TokenizerWorker(Process):
    def __init__(self, worker_id: int, input_queue: Queue, output_queue: Queue, tokenizer):
        super().__init__()
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.tokenizer = tokenizer
        self.chunk_num = 4

    def run(self):
        while True:
            try:
                batch = self.input_queue.get(timeout=60)
                if batch is None:  # Stop signal
                    break

                texts = batch['text']
                indices = [idx.item() if torch.is_tensor(idx) else idx for idx in batch['index']]

                all_chunked_texts = []
                all_chunk_indices = []

                # Tokenize and split texts into chunks
                for i, text in enumerate(texts):
                    tokens = self.tokenizer.tokenize(text)
                    step_size = len(tokens) // self.chunk_num if len(tokens) >= self.chunk_num else 1  # Evenly split into 8 chunks

                    chunked_texts = [self.tokenizer.convert_tokens_to_string(tokens[j * step_size: (j + 1) * step_size]) for j in range(self.chunk_num) if len(tokens[j * step_size: (j + 1) * step_size]) > 0]

                    all_chunked_texts.extend(chunked_texts)
                    all_chunk_indices.extend([indices[i]] * len(chunked_texts))

                self.output_queue.put({
                    'chunked_texts': all_chunked_texts,
                    'chunk_indices': all_chunk_indices
                })

            except queue.Empty:
                continue


def embeder(model, data_path, save_path):

    dataset = SentenceDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=2000, shuffle=False)

    # Create queues and multiple worker processes
    input_queue = Queue(maxsize=100)
    output_queue = Queue(maxsize=100)
    
    # Create a thread to feed data into the queue
    def fill_queue():
        try:
            for batch in dataloader:
                # Convert tensor to normal Python types
                if 'text' in batch:
                    batch_dict = {
                        'text': batch['text'],
                        'index': [idx.item() if torch.is_tensor(idx) else idx for idx in batch['index']]
                    }
                input_queue.put(batch_dict)
            # Send a termination signal to each worker
            for _ in range(NUM_TOKENIZER_WORKERS):
                input_queue.put(None)
        except Exception as e:
            print(f"Error in fill_queue: {e}")

    fill_thread = threading.Thread(target=fill_queue)
    fill_thread.start()
    
    workers = []
    for i in range(NUM_TOKENIZER_WORKERS):
        worker = TokenizerWorker(i, input_queue, output_queue, model.tokenizer)
        worker.start()
        workers.append(worker)

    # Main process handles the results
    encoded_sentences = {}
    processed_batches = 0
    total_batches = len(dataloader)
    with tqdm.tqdm(total=total_batches) as pbar:
        while processed_batches < total_batches:
            try:
                result = output_queue.get(timeout=300)  # Increase timeout to 5 minutes

                all_chunked_texts = result['chunked_texts']
                all_chunk_indices = result['chunk_indices']

                # Encoding process
                if all_chunked_texts:
                    embeddings = model.encode(
                        all_chunked_texts,
                        batch_size=len(all_chunked_texts),
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                    sentence_embeddings = {}
                    # Aggregate embeddings of the same sentence
                    for idx, embedding in zip(all_chunk_indices, embeddings):
                        idx_key = idx if isinstance(idx, int) else idx.item()
                        if idx_key not in sentence_embeddings:
                            sentence_embeddings[idx_key] = []
                        sentence_embeddings[idx_key].append(embedding)
                                
                    # Use index as key when storing
                    for idx, emb_list in sentence_embeddings.items():
                        avg_embedding = torch.mean(torch.stack(emb_list), dim=0)
                        encoded_sentences[idx] = avg_embedding.tolist()

                processed_batches += 1
                pbar.update(1)

            except queue.Empty:
                print("Timeout waiting for results. Check if worker is still processing.")
                continue
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    # Clear CUDA cache
    torch.cuda.empty_cache()
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx in sorted(encoded_sentences.keys()):  # Numeric sorting is fast
            json_line = json.dumps({
                'index': idx, 
                'embedding': encoded_sentences[idx]
            }, ensure_ascii=False)
            f.write(json_line + '\n')
            
# if __name__ == "__main__":
#     embeder(model, data_path, result_path)
