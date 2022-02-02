from .utils import Opt
from .transformer_layers import Transformer

import torch
import torch.nn as nn

import os


def load_model():
    """The function used to load the model's parameters"""
    opt = Opt.get_instance()

    model = Transformer(src_vocab_size=opt.vocab_size,
                        trg_vocab_size=opt.vocab_size,
                        model_dim=opt.model_dim,
                        d_ff=opt.model_dim * 4,
                        heads=opt.heads,
                        num_blocks=opt.N,
                        max_seq_len=opt.max_len,
                        norm_before=True).to(opt.device)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    starting_index = 0
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # initializing the parameters of the model.
    if not os.path.exists(opt.path):
        if not os.path.exists(opt.path):
            os.mkdir(opt.path)
        opt.log.print(f"No {opt.path} found. Created a new path directory and started using xavier_uniform")
    else:
        for i in os.walk(opt.path):
            break
        i = i[2]
        m = 0
        mf = None
        suffix_len = len('.model')
        for file in i:
            if opt.model_prefix not in file or '.model' not in file:
                continue

            num = int(file[len(opt.model_prefix):-suffix_len])
            if num > m:
                m = num
                mf = file[:-suffix_len]

        if mf is not None:
            opt.log.print(f"Starting from last saved {mf}")
            opt.save_model.load(f'{opt.path}/{mf}')
            model.load_state_dict(opt.save_model.model_state_dict)
            optim.load_state_dict(opt.save_model.optim_state_dict)
            starting_index = m
        else:
            opt.log.print(f"Starting from xavier_uniform distribution")
    opt.starting_index = starting_index

    opt.model = model
    opt.optim = optim
    # return model, optim