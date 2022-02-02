from dataclasses import dataclass
from typing import ClassVar

import torch
from torch.autograd import Variable
from datetime import datetime
import numpy as np
Transformer = torch.nn.Transformer


class Log:
    """
    A logger that notes the date/time with the text provided.
    """

    LOG, ERROR = 0, 1

    def __init__(self, outfile='data/.log', filename='data/logfile.log'):
        self.filename = filename
        self.outfile = outfile
        self.file_object = open(filename, 'a+', encoding='utf-8')
        self.line_num = 0
        print("LOGGING For seesion on: " + str(datetime.now()), file=self.file_object)

    def print(self, txt, type=LOG, shell=True):
        if shell: print(txt)
        prefix = "LOG ::" if type==Log.LOG else "ERROR ::"
        txt = f"{prefix} {str(datetime.now())} :: {txt}"
        print(txt, file=self.file_object)

    def close(self):
        self.file_object.seek(0, 0)
        text = self.file_object.read()
        text = text.split('\n')
        text.reverse()
        output = '\n'.join(text)
        self.file_object.close()

        with open(self.outfile, 'w') as f:
            f.write(output)

    def flush(self):
        self.file_object.close()
        self.file_object = open(self.filename, 'a+', encoding = 'utf-8')


def batch():
    """The batching generator"""
    opt = Opt.get_instance()
    max_count = {v: (len(opt.src_bins[v])*v) // opt.tokensize for v in opt.bins}
    # print(max_count)
    cur_count = {v: 0 for v in opt.bins}
    batch_sizes = {v: opt.tokensize // v for v in opt.bins}

    step = 0
    while len(max_count):
        v = np.random.choice(list(max_count.keys()))
        i = cur_count[v]
        cur_count[v] += 1
        j = cur_count[v]

        step += 1
        if step < opt.starting_index:
            continue

        size = batch_sizes[v]
        src_list = opt.src_bins[v][i*size: j*size]
        trg_list = opt.trg_bins[v][i*size: j*size]

        if j > max_count[v]:
            if opt.keep_training:
                cur_count[v] = 0
            else:
                del max_count[v]

        if len(src_list) == 0:
            continue

        yield src_list, trg_list


def nopeak_mask(size):
    """The function which generates an upper triangular matrix"""
    opt = Opt.get_instance()
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0).to(opt.device)
    return np_mask


def create_masks(src, trg):
    """
    The function that makes the source and target mask
    :param src: The source batch
    :param trg: The target batch
    :return: the source and target mask
    """
    opt = Opt.get_instance()
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size).to(opt.device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


class SingletonException(Exception):
    def __init__(self, message):
        super().__init__(message)


class Save:
    '''
    A utility class that is used to save and load the parameters into the model and optimizer
    '''
    def __init__(self, model_state_dict=None, optim_state_dict=None):
        self.model_state_dict = model_state_dict
        self.optim_state_dict = optim_state_dict

    def save(self, filename):
        torch.save(self.model_state_dict, f'{filename}.model')
        torch.save(self.optim_state_dict, f'{filename}.optim')

    def load(self, filename):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        self.model_state_dict = torch.load(f'{filename}.model', map_location=device)
        self.optim_state_dict = torch.load(f'{filename}.optim', map_location=device)


@dataclass
class Opt:
    __instance: ClassVar = None

    # Enter the rest of the parameters here:

    src_lang: str = ''
    trg_lang: str = ''
    dir_name: str = ''

    k: int = 10
    model_num: int = 1000 * 120

    num_mil: int = 1
    max_len = 150

    vocab_size: int = 8000
    tokensize: int = 4096
    print_every: int = 200
    save_every: int = 5000
    epochs: int = 10
    warmup_steps: int = 16000
    keep_training: True = False
    starting_index: int = 0
    step: int = 0

    save_model: Save = Save()

    data_path: str = '/content/drive/MyDrive/Dissertation/data'

    eval_path: str = '/content/drive/MyDrive/Dissertation/eval/'

    model_dim: int = 512
    heads: int = 8
    N: int = 6

    model: torch.nn.Module = None
    optim: torch.optim.Adam = None


    @property
    def model_file(self):
        # the sentence-piece model file prefix
        return f'../data/{self.dir_name}/SPM-8k.{self.dir_name}.'


    @property
    def input_file(self):
        # the constructed dataset file
        return f'../data/{self.src_lang}/{self.dir_name}.'


    @property
    def dataset(self):
        # the tokenized and binned dataset
        return f'../data/{self.dir_name}/tokenized_dataset_{self.dir_name}'


    @property
    def dev_dataset(self):
        # the evaluation dataset used at regular intervals in the training
        return f'../data/{self.dir_name}/DEV-{self.dir_name}.'


    @property
    def src_data_path(self):
        return self.input_file + self.src_lang


    @property
    def trg_data_path(self):
        return self.input_file + self.trg_lang

    @property
    def path(self):
        return f"../data/{self.dir_name}/{self.src_lang}-{self.trg_lang}-models"

    @property
    def dev_src_path(self):
        return f"../data/{self.dir_name}/dev-{self.src_lang}"

    @property
    def dev_trg_path(self):
        return f"../data/{self.dir_name}/dev-{self.trg_lang}"

    # End of parameters:

    def __post_init__(self):
        if Opt.__instance is not None:
            raise SingletonException("Singleton Class can only have one instance")
        else:
            Opt.__instance = self

        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'

        self.device: torch.device = torch.device(dev)

    @staticmethod
    def get_instance():
        if Opt.__instance is None:
            Opt()

        return Opt.__instance
