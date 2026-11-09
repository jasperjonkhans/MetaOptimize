"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
import hashlib
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_shards(data_dir, vocab_source, vocab_size):
    directory = Path(data_dir) / ("TinyStories_all_data" if vocab_source == "llama2" else f"tok{vocab_size}")
    return sorted(directory.glob("*.bin"))


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(p for p in glob.glob(os.path.join(data_dir, "*.json"))
                             if os.path.basename(p) != "manifest.json")
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")

def train_vocab(vocab_size):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 10

    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(p for p in glob.glob(os.path.join(data_dir, "*.json"))
                             if os.path.basename(p) != "manifest.json")

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(input=tiny_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity")

    # 3) optional cleanup, ask the user if they'd like to delete tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def process_shard(args, vocab_size, data_dir=DATA_CACHE_DIR, tokenizer_model=None):
    shard_id, shard = args
    tokenizer_model = tokenizer_model or get_tokenizer_model_path(vocab_size, data_dir)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    if not all_tokens or min(all_tokens) < 0 or max(all_tokens) > 65535:
        raise ValueError(f"Empty shard or token outside uint16 range: {shard}")
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(data_dir, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    temporary = tokenized_filename + ".tmp"
    with open(temporary, "wb") as f:
        f.write(all_tokens.tobytes())
    os.replace(temporary, tokenized_filename)
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")
    return {"file": os.path.basename(tokenized_filename), "tokens": int(all_tokens.size),
            "sha256": file_sha256(tokenized_filename), "source_sha256": file_sha256(shard)}


def pretokenize(vocab_size, data_dir=DATA_CACHE_DIR, tokenizer_model=None, max_workers=None):
    # iterate the shards and tokenize all of them one by one
    source_dir = Path(data_dir) / "TinyStories_all_data"
    shard_filenames = sorted(str(p) for p in source_dir.glob("*.json") if p.name != "manifest.json")
    if not shard_filenames:
        raise ValueError(f"No JSON shards in {source_dir}")
    bin_dir = source_dir if vocab_size == 0 else Path(data_dir) / f"tok{vocab_size}"
    bin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bin_dir / "manifest.json"
    # A failed rerun must never leave an old success manifest behind.
    manifest_path.unlink(missing_ok=True)
    tokenizer_model = tokenizer_model or get_tokenizer_model_path(vocab_size, data_dir) or "tokenizer.model"
    fun = partial(process_shard, vocab_size=vocab_size, data_dir=data_dir,
                  tokenizer_model=tokenizer_model)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fun, enumerate(shard_filenames)))
    expected = {Path(p).with_suffix(".bin").name for p in shard_filenames}
    if ({r["file"] for r in results} != expected or len(results) != len(expected)
            or {p.name for p in bin_dir.glob("*.bin")} != expected):
        raise ValueError("Incomplete or unexpected token shard set")
    for result in results:
        path = bin_dir / result["file"]
        if path.stat().st_size != 2 * result["tokens"] or file_sha256(path) != result["sha256"]:
            raise ValueError(f"Invalid token shard: {path}")
    manifest = {"version": 1, "vocab_size": vocab_size,
                "tokenizer_sha256": file_sha256(tokenizer_model), "shards": results}
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    print("Done.")
    return manifest


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source,
                 seed=1, data_dir=DATA_CACHE_DIR, rank=None):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        self.seed = seed
        self.data_dir = data_dir
        self.rank = dist.get_rank() if rank is None and dist.is_initialized() else (rank or 0)
        if split not in {"train", "val"} or vocab_source not in {"llama2", "custom"}:
            raise ValueError("Invalid split or vocab source")

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = self.rank
        # Stable streams per CLI seed, process rank, and DataLoader worker.
        seed = self.seed + worker_id + 1000003 * rank
        rng = random.Random(seed)
        shard_filenames = token_shards(self.data_dir, self.vocab_source, self.vocab_size)
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        if not shard_filenames:
            raise ValueError(f"No {self.split} bin shards in {self.data_dir}")
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y

# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size, data_dir=DATA_CACHE_DIR):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(data_dir, f"tok{vocab_size}.model")

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        generator = torch.Generator().manual_seed(ds.seed + ds.rank)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=str(device).startswith("cuda"),
            num_workers=num_workers, worker_init_fn=seed_worker, generator=generator
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--data-dir", default=DATA_CACHE_DIR)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()
    DATA_CACHE_DIR = args.data_dir

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size, data_dir=args.data_dir,
                    tokenizer_model=args.tokenizer_model, max_workers=args.max_workers)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
