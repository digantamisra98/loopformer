"""Tiny prototype dataset (GPT-2 BPE tinyshakespeare) for fast loopformer iteration.
Produces train.bin / val.bin (uint16). For scale, swap dataset to fineweb_edu/openwebtext."""
import os
import requests
import numpy as np
import tiktoken

here = os.path.dirname(__file__)
input_file_path = os.path.join(here, 'input.txt')
if not os.path.exists(input_file_path):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data, val_data = data[:int(n * 0.9)], data[int(n * 0.9):]

enc = tiktoken.get_encoding("gpt2")
np.array(enc.encode_ordinary(train_data), dtype=np.uint16).tofile(os.path.join(here, 'train.bin'))
np.array(enc.encode_ordinary(val_data), dtype=np.uint16).tofile(os.path.join(here, 'val.bin'))
print("shakespeare prepared:", os.path.join(here, 'train.bin'))
