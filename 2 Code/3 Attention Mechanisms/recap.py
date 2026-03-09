import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):

    def __init__(self, text, tokenizer, context_size=256, stride=32):
        super().__init__()

        self.tokenizer = tokenizer
        self.input = []
        self.output = []
        
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for id_idx in range(0, len(token_ids) - context_size, stride):
            inputs = token_ids[id_idx : id_idx + context_size]
            outputs = token_ids[id_idx+1 : id_idx + context_size + 1]
            self.input.append(inputs)
            self.output.append(outputs)

    def __getitem__(self, index): 
        return torch.tensor(self.input[index]), torch.tensor(self.output[index])

    def __len__(self):
        return len(self.input)


###

import torch.nn as nn

class Embeddings(nn.Module):

    def __init__(self, vocab_size, context_size, emb_dim, padding_idx=-100):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx)
        self.pos_emb = nn.Embedding(context_size, emb_dim)

    def forward(self, x):

        batch_size, seq_len = x.shape

        return self.word_emb(x) + self.pos_emb(torch.arange(seq_len))