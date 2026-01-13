import json
import os
import torch
import tqdm
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer


def _read_last_complete_jsonl_line(path: str, max_read_bytes: int = 1024 * 1024) -> Optional[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    size = os.path.getsize(path)
    read_size = min(size, max_read_bytes)

    with open(path, "rb") as f:
        f.seek(size - read_size)
        data = f.read(read_size)

    first_nl = data.find(b"\n")
    if first_nl != -1 and size > read_size:
        data = data[first_nl + 1 :]

    lines = data.split(b"\n")

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        try:
            line_str = line.decode("utf-8")
            json.loads(line_str)
            return line_str
        except Exception:
            continue

    return None


def get_last_done_index_and_fix_file(path: str) -> int:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0

    last_line = _read_last_complete_jsonl_line(path)
    if last_line is None:
        with open(path, "wb") as f:
            f.truncate(0)
        return 0

    last_obj = json.loads(last_line)
    last_idx = int(last_obj["index"])

    size = os.path.getsize(path)
    read_size = min(size, 1024 * 1024)

    with open(path, "rb") as f:
        f.seek(size - read_size)
        data = f.read(read_size)

    last_bytes = (last_line + "\n").encode("utf-8")
    pos = data.rfind(last_bytes)
    if pos != -1:
        start_of_chunk = size - read_size
        truncate_pos = start_of_chunk + pos + len(last_bytes)
        with open(path, "rb+") as f:
            f.truncate(truncate_pos)

    return last_idx


class SentenceDataset(Dataset):
    def __init__(self, data_path: str, start_index: int = 1):
        self.data_path = data_path
        self.offsets: List[int] = []
        self._build_offsets()


        self.start_pos = max(0, start_index - 1)

    def _build_offsets(self):
        offset = 0
        with open(self.data_path, "rb") as f:
            for line in f:
                self.offsets.append(offset)
                offset += len(line)

    def __len__(self):
        return max(0, len(self.offsets) - self.start_pos)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        real_idx = self.start_pos + i 
        with open(self.data_path, "rb") as f:
            f.seek(self.offsets[real_idx])
            line = f.readline().decode("utf-8")

        obj = json.loads(line)
        if "text" in obj:
            text = obj["text"]
        else:
            text = str(obj.get("prompt", "")) + str(obj.get("response", ""))

        return {"index": real_idx + 1, "text": text} 


# =====================
# Worker-local tokenizer
# =====================
_WORKER_TOKENIZER = None

def _get_worker_tokenizer(model: SentenceTransformer):
    global _WORKER_TOKENIZER
    if _WORKER_TOKENIZER is None:
        _WORKER_TOKENIZER = model.tokenizer
    return _WORKER_TOKENIZER


def make_collate_fn(model: SentenceTransformer, chunk_num: int):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenizer = _get_worker_tokenizer(model)

        all_chunked_texts: List[str] = []
        all_chunk_indices: List[int] = []

        for item in batch:
            idx = int(item["index"])
            text = item["text"]

            tokens = tokenizer.tokenize(text)
            if not tokens:
                continue

            step_size = len(tokens) // chunk_num if len(tokens) >= chunk_num else 1

            for j in range(chunk_num):
                start = j * step_size
                end = (j + 1) * step_size
                if start >= len(tokens):
                    break
                chunk_tokens = tokens[start:end]
                if not chunk_tokens:
                    continue
                chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
                if chunk_text.strip():
                    all_chunked_texts.append(chunk_text)
                    all_chunk_indices.append(idx)

        return {"chunked_texts": all_chunked_texts, "chunk_indices": all_chunk_indices}

    return collate_fn


def embeder(
    model: SentenceTransformer,
    data_path: str,
    save_path: str,
    batch_size: int,
    chunk_num: int,
    num_workers: int = 16,
    resume: bool = True,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    last_done = 0
    if resume and os.path.exists(save_path):
        last_done = get_last_done_index_and_fix_file(save_path)

    start_index = last_done + 1
    dataset = SentenceDataset(data_path, start_index=start_index)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=make_collate_fn(model, chunk_num),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    mode = "a" if (resume and last_done > 0) else "w"
    with open(save_path, mode, encoding="utf-8") as f_out:
        with tqdm.tqdm(total=len(dataloader), desc=f"Embedding (start={start_index})") as pbar:
            for batch in dataloader:
                chunked_texts = batch["chunked_texts"]
                chunk_indices = batch["chunk_indices"]

                if not chunked_texts:
                    pbar.update(1)
                    continue

                embeddings = model.encode(
                    chunked_texts,
                    batch_size=len(chunked_texts),
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )

                per_sent: Dict[int, List[torch.Tensor]] = {}
                for idx, emb in zip(chunk_indices, embeddings):
                    per_sent.setdefault(int(idx), []).append(emb)

                for idx in sorted(per_sent.keys()):
                    avg_emb = torch.mean(torch.stack(per_sent[idx], dim=0), dim=0)
                    f_out.write(json.dumps({"index": idx, "embedding": avg_emb.tolist()}, ensure_ascii=False) + "\n")

                pbar.update(1)

    torch.cuda.empty_cache()

