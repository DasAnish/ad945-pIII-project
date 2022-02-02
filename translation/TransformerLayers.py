import torch
import torch.nn as nn
from typing import Optional, List
from .SubLayers import PositionalEncoding, MultiHeadAttention, FeedForward
Tensor = torch.Tensor


class Transformer(nn.Module):
    """The implementation of the Transformer architecture from the "Attention is all you need" paper"""
    def __init__(self, src_vocab_size: int, trg_vocab_size: int,
                 model_dim: int, d_ff: int, heads: int, num_blocks: int,
                 max_seq_len: Optional[int] = 80, dropout: Optional[float] = 0.1,
                 norm_before: bool = True):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.encoder: TransformerEncoder = TransformerEncoder(model_dim, d_ff, heads, num_blocks,
                                                              src_vocab_size, max_seq_len, dropout,
                                                              norm_before).to(device)
        self.decoder: TransformerDecoder = TransformerDecoder(model_dim, d_ff, heads, num_blocks,
                                                              trg_vocab_size, max_seq_len, dropout,
                                                              norm_before).to(device)

        self.linear: nn.Linear = nn.Linear(model_dim, trg_vocab_size).to(device)

        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)
        self.add_module('linear', self.linear)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, trg_mask: Tensor):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)
        output = self.linear(dec_output)
        del enc_output, dec_output
        torch.cuda.empty_cache()
        return output


class TransformerEncoderLayer(nn.Module):
    """The class implementing a single encoder layer in the Transformer architecture."""
    def __init__(self, model_dim: int, heads: int, d_ff: int,
                 dropout: Optional[float] = 0.1, norm_before: bool = False):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.dropout: float = dropout
        self.norm_before: bool = norm_before

        self.self_attn: MultiHeadAttention = MultiHeadAttention(model_dim,
                                                                heads,
                                                                dropout).to(device)
        self.add_module('self_attn', self.self_attn)

        self.attn_dropout: nn.Dropout = nn.Dropout(dropout).to(device)
        self.ffn_dropout: nn.Dropout = nn.Dropout(dropout).to(device)
        self.add_module('attn_dropout', self.attn_dropout)
        self.add_module('ffn_dropout', self.ffn_dropout)

        self.ffn: FeedForward = FeedForward(model_dim, d_ff, dropout).to(device)
        self.add_module('ffn', self.ffn)

        self.attn_norm: nn.LayerNorm = nn.LayerNorm(model_dim).to(device)
        self.ffn_norm: nn.LayerNorm = nn.LayerNorm(model_dim).to(device)
        self.add_module('attn_norm', self.attn_norm)
        self.add_module('ffn_norm', self.ffn_norm)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        y = x
        if self.norm_before:
            y = self.attn_norm(x)
        # print('encoder_attention')
        y = self.self_attn(y, y, y, mask)
        y = self.attn_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.attn_norm(x)

        y = x
        if self.norm_before:
            y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.ffn_norm(x)

        return x


class TransformerEncoder(nn.Module):
    """The class that is a collection of Encoder Layers"""
    def __init__(self, model_dim: int, d_ff: int, heads: int, num_blocks: int,
                 vocab_size: int, max_seq_len: Optional[int] = 80,
                 dropout: Optional[float] = 0.1, norm_before: bool = False):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.num_blocks: int = num_blocks
        self.vocab_size: int = vocab_size
        self.max_seq_len: int = max_seq_len

        self.word_embeddings: nn.Embedding = nn.Embedding(vocab_size, model_dim).to(device)
        self.positional_encoding: PositionalEncoding = PositionalEncoding(model_dim, max_seq_len).to(device)
        self.add_module('word_embeddings', self.word_embeddings)

        args = (model_dim, heads, d_ff, dropout, norm_before)
        # encoding_layer: TransformerEncoderLayer = TransformerEncoderLayer(*args).to(device)
        self.encoding_layers: List[TransformerEncoderLayer] = [TransformerEncoderLayer(*args).to(device)
                                                               for _ in range(num_blocks)]
        for i, layer in enumerate(self.encoding_layers):
            self.add_module(f'layer_{i}', layer)

    def forward(self, _input: Tensor, mask: Tensor):
        x = self.word_embeddings(_input)
        x = self.positional_encoding(x)
        # print(f"x.shape {x.shape}")
        for layer in self.encoding_layers:
            x = layer(x, mask)

        return x


class TransformerDecoderLayer(nn.Module):
    """The class implementing a single decoder layer in the Transformer architecture."""
    def __init__(self, model_dim: int, heads: int, d_ff: int,
                 dropout: Optional[float] = 0.1, norm_before: bool = False):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.d_ff: int = d_ff
        self.dropout: float = dropout
        self.norm_before: bool = norm_before

        self.self_attn_norm: nn.LayerNorm = nn.LayerNorm(model_dim).to(device)
        self.enc_dec_norm: nn.LayerNorm = nn.LayerNorm(model_dim).to(device)
        self.ffn_norm: nn.LayerNorm = nn.LayerNorm(model_dim).to(device)
        self.add_module('self_attn_norm', self.self_attn_norm)
        self.add_module('enc_dec_norm', self.enc_dec_norm)
        self.add_module('ffn_norm', self.ffn_norm)

        self.self_attn_dropout: nn.Dropout = nn.Dropout(dropout).to(device)
        self.enc_dec_dropout: nn.Dropout = nn.Dropout(dropout).to(device)
        self.ffn_dropout: nn.Dropout = nn.Dropout(dropout).to(device)
        self.add_module('self_attn_dropout', self.self_attn_dropout)
        self.add_module('enc_dec_dropout', self.self_attn_dropout)
        self.add_module('ffn_dropout', self.ffn_dropout)

        self.self_attn: MultiHeadAttention = MultiHeadAttention(model_dim, heads, dropout).to(device)
        self.enc_dec_attn: MultiHeadAttention = MultiHeadAttention(model_dim, heads, dropout).to(device)
        self.ffn: FeedForward = FeedForward(model_dim, d_ff).to(device)
        self.add_module('self_attn', self.self_attn)
        self.add_module('enc_dec_attn', self.enc_dec_attn)
        self.add_module('ffn', self.ffn)

    def forward(self, x: Tensor, encoder_output: Tensor, src_mask: Tensor, trg_mask: Tensor) -> Tensor:

        y = x
        if self.norm_before:
            y = self.self_attn_norm(x)
        # print('decoder_attention')
        y = self.self_attn(y, y, y, trg_mask)
        y = self.self_attn_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.self_attn_norm(x)

        y = x
        if self.norm_before:
            y = self.enc_dec_norm(x)
        # print('enc_dec_attn')
        y = self.enc_dec_attn(y, encoder_output, encoder_output,
                              src_mask)
        y = self.enc_dec_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.enc_dec_norm(x)

        y = x
        if self.norm_before:
            y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.ffn_norm(x)

        return x


class TransformerDecoder(nn.Module):
    """The class that is a collection of Decoder Layers"""
    def __init__(self, model_dim: int, d_ff: int, heads: int, num_blocks: int,
                 vocab_size: int, max_seq_len: Optional[int] = 80,
                 dropout: Optional[float] = 0.1, norm_before: bool = True):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.num_blocks: int = num_blocks
        self.vocab_size: int = vocab_size
        self.max_seq_len: int = max_seq_len
        self.dropout: float = dropout
        self.norm_before: bool = norm_before

        self.word_embeddings: nn.Embedding = nn.Embedding(vocab_size, model_dim).to(device)
        self.positional_embeddings: PositionalEncoding = PositionalEncoding(model_dim, max_seq_len).to(device)

        args = (model_dim, heads, d_ff, dropout, norm_before)

        # decoding_layer: TransformerDecoderLayer = TransformerDecoderLayer(*args).to(device)
        self.decoding_layers: List[TransformerDecoderLayer] = [TransformerDecoderLayer(*args).to(device)
                                                               for _ in range(num_blocks)]
        for i, layer in enumerate(self.decoding_layers):
            self.add_module(f'layer_{i}', layer)

    def forward(self, target: Tensor, encoder_output: Tensor, src_mask: Tensor, trg_mask: Tensor) -> Tensor:
        x: Tensor = self.word_embeddings(target)
        x = self.positional_embeddings(x)

        for layer in self.decoding_layers:
            x = layer(x, encoder_output, src_mask, trg_mask)

        return x