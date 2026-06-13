"""Streaming fineweb-edu prep -> train.bin/val.bin (uint16, build_loader-compatible).
Streams (no full download) and tokenizes up to TOK_BUDGET tokens. A large real corpus for
the looped experiments (vs the toy shakespeare prototype). Scale by raising TOK_BUDGET.

Env: TOK_BUDGET (default 600M), OUT_DIR (default this dir), FW_NAME (default sample-10BT)."""
import os
import numpy as np
import tiktoken
from datasets import load_dataset

BUDGET = int(os.environ.get("TOK_BUDGET", 600_000_000))
OUT = os.environ.get("OUT_DIR", os.path.dirname(__file__))
NAME = os.environ.get("FW_NAME", "sample-10BT")
os.makedirs(OUT, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token
ds = load_dataset("HuggingFaceFW/fineweb-edu", name=NAME, split="train", streaming=True)

val_budget = BUDGET // 50
n = 0
tf = open(os.path.join(OUT, "train.bin"), "wb")
vf = open(os.path.join(OUT, "val.bin"), "wb")
for ex in ds:
    ids = enc.encode_ordinary(ex["text"])
    ids.append(eot)
    arr = np.array(ids, dtype=np.uint16)
    (vf if n >= BUDGET - val_budget else tf).write(arr.tobytes())
    n += len(ids)
    if n % 50_000_000 < len(ids):
        print(f"  {n/1e6:.0f}M tokens", flush=True)
    if n >= BUDGET:
        break
tf.close(); vf.close()
print(f"done: {n} tokens -> {OUT}/train.bin,val.bin")
