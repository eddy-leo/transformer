import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoConfig

model_ckpt = 'google-bert/bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
text = "Hello world"
inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
print(inputs)

config = AutoConfig.from_pretrained(model_ckpt)
token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_embed)

input_embed = token_embed(inputs['input_ids'])
print(input_embed)
