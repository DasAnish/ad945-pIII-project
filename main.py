import torch
import torch.nn as nn
import torch.nn.functional as F

from translation.utils import Opt, Log
from translation.preprocess_data import *
from translation.train_model import *


if __name__ == '__main__':
    opt = Opt.get_instance()
    opt.lang1 = 'hi'
    opt.lang2 = 'mr'

    print(opt.interim_file)