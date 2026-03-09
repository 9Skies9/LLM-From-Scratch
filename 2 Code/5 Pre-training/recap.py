import torch
from torch.utils.data import Dataset, DataLoader
from math import sqrt

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
    

###

class Multi_Head_Attention(nn.Module):

    def __init__(self, hidden_dim, context_size, dropout, head_nums):
        super().__init__()

        self.k_weights = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q_weights = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_weights = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head_size = int(hidden_dim/head_nums)
        self.head_nums = head_nums

        self.dropout = nn.Dropout(dropout)
        self.mask = torch.triu(torch.ones((context_size, context_size)), diagonal=1).bool()

        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)


    def forward(self, x, casual):

        # x size [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape

        q = self.q_weights(x)
        k = self.k_weights(x)
        v = self.v_weights(x)

        # result shape [batch_size, num_heads, seq_len, head_size] <- [batch_size, seq_len, num_heads, head_size]
        k = k.view(batch_size, seq_len, self.head_nums, self.head_size).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.head_nums, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.head_nums, self.head_size).transpose(1, 2)

        # attn_result size [batch_size, head_nums, seq_len, seq_len]
        attn_result = q @ k.transpose(-2, -1)

        if casual == True:
            mask = self.mask[:seq_len, :seq_len]
            attn_values = torch.softmax(torch.masked_fill(attn_result, mask, -torch.inf) / sqrt(self.head_size), dim=-1)
        else:
            attn_values = torch.softmax(attn_values / sqrt(self.head_size), dim=-1)

        # result shape [batch_size, seq_len, num_heads, head_size] <- [batch_size, num_heads, seq_len, head_size]
        result = (self.dropout(attn_result) @ v).transpose(1, 2)

        # result shape [batch_size, seq_len, num_heads * head_size]
        result = self.mlp(result.contiguous().view(batch_size, seq_len, self.head_nums * self.head_size))

        return result, attn_values
    

###

import torch
import torch.nn as nn
from recap import Multi_Head_Attention


class Transformer_Block(nn.Module):

    def __init__(self, hidden_dim, context_size, dropout, head_nums):
        super().__init__()

        self.attn = Multi_Head_Attention(hidden_dim, context_size, dropout, head_nums)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, casual):
        
        x_res = x

        x = self.layer_norm_1(x)
        x, _ = self.attn(x, casual)
        x = self.dropout(x) + x_res

        x_res = x

        x = self.layer_norm_2(x)
        x = self.mlp(x)
        x = self.dropout(x) + x_res

        return x


###


class GPT_Model(nn.Module):

    def __init__(self, hidden_dim, output_dim, context_size, dropout, head_nums, num_layers):
        super().__init__()

        self.emb_layer = Embeddings(vocab_size=output_dim, context_size=context_size, emb_dim=hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [Transformer_Block(hidden_dim, context_size, dropout, head_nums)
             for _ in range(num_layers)]
        )
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x, casual):

        x = self.emb_dropout(self.emb_layer(x))
        
        for block in self.transformer_blocks:
            x = block(x, casual)

        x = self.output_layer(x)

        return x