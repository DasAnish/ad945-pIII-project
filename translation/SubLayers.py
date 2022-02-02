import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, sin, cos
from typing import Optional
Tensor = torch.Tensor
Variable = torch.autograd.Variable


class PositionalEncoding(nn.Module):

    """A parameter less module that concatenates a number of sine signals at the end of the embedded vectors."""

    def __init__(self, model_dim: int, max_length: int, dropout: float = 0.1):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim: int = model_dim
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.add_module('dropout', self.dropout)

        position_vector: Tensor = torch.zeros(max_length, model_dim, requires_grad=False).to(device)

        # note to self: this appear to work right now.
        for pos in range(max_length):
            for i in range(0, model_dim, 2):
                # Follwing the formula provided in the paper.
                position_vector[pos, i] = sin(pos / (10000 ** ((2 * i) / model_dim)))
                position_vector[pos, i+1] = cos(pos / (10000 ** ((2 * (i + 1)) / model_dim)))

        # position_vector: max_seq_len x model_dim
        position_vector = position_vector.unsqueeze(0)

        # position_vector: 1 x max_seq_len x model_dim
        self.register_buffer('position_vector', position_vector)

    def forward(self, x: Tensor) -> Tensor:

        x = x * sqrt(self.model_dim)
        sequence_length = x.size(1)
        x = x + Variable(self.position_vector[:, :sequence_length], requires_grad=False)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    The nn.Module that implements the Multihead attention operation
    For more information see the paper "Attention is all you need"
    """

    def __init__(self, model_dim: int, heads: int, dropout: float = 0.1):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.d_k: int = model_dim // heads
        self.dropout: nn.Dropout = nn.Dropout(dropout).to(device)
        self.add_module('dropout', self.dropout)

        self.w_key: nn.Linear = nn.Linear(model_dim, model_dim).to(device)
        self.w_query: nn.Linear = nn.Linear(model_dim, model_dim).to(device)
        self.w_value: nn.Linear = nn.Linear(model_dim, model_dim).to(device)
        self.add_module('w_key', self.w_key)
        self.add_module('w_query', self.w_query)
        self.add_module('w_value', self.w_value)

        self.final_layer: nn.Linear = nn.Linear(model_dim, model_dim).to(device)
        self.add_module('final_layer', self.final_layer)

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Optional[Tensor] = None):

        # if encoder_output is not None: print("enc_output == True")
        # print(f'x.shape: {x.shape}')

        batch_size = q.size(0)
        queries = self.w_query(q).view(batch_size, -1, self.heads, self.d_k)
        keys = self.w_key(k).view(batch_size, -1, self.heads, self.d_k)
        values = self.w_value(v).view(batch_size, -1, self.heads, self.d_k)

        # we have batch_size, seq_len, heads, d_k tensor

        keys.transpose_(1, 2)
        queries.transpose_(1, 2)
        values.transpose_(1, 2)

        interim_result = torch.matmul(queries, keys.transpose(-2, -1)) / sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            interim_result = interim_result.masked_fill(mask == 0, -1e9)

        interim_result = F.softmax(interim_result, dim=-1)

        if self.dropout is not None:
            interim_result = self.dropout(interim_result)

        interim_result = torch.matmul(interim_result, values)

        concatenated_result = interim_result.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)

        final_result = self.final_layer(concatenated_result)

        return final_result


class FeedForward(nn.Module):

    """The utility module that is used to implement a feed-forward fully connected Neural Net for the
    encoder and decoder layers."""

    def __init__(self, model_dim: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.fc1: nn.Linear = nn.Linear(model_dim, d_ff).to(device)
        self.dropout: nn.Dropout = nn.Dropout(dropout).to(device)
        self.fc2: nn.Linear = nn.Linear(d_ff, model_dim).to(device)

        self.add_module('fc1', self.fc1)
        self.add_module('dropout', self.dropout)
        self.add_module('fc2', self.fc2)

    def forward(self, x: Tensor) -> Tensor:

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x