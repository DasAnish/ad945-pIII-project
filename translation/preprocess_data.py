from translation.utils import  Opt
import pickle
from tqdm.notebook import tnrange
import sentencepiece as spm
import os


def move():
    """The function that is used to construct the dataset"""
    opt = Opt.get_instance()

    def move_lang(lang):
        inpFile1 = open(opt.pc_input_file + lang, 'r', encoding='utf-8')
        inpFile2 = open(opt.nc_input_file + lang, 'r', encoding='utf-8')
        intFile = open(opt.interim_file + lang, 'w', encoding='utf-8')

        for _ in tnrange(int(opt.num_mil * 2 * 10**5)):
            intFile.write(inpFile2.readline())
        for _ in tnrange(int(opt.num_mil * 8 * 10**5)):
            intFile.write(inpFile1.readline())

        inpFile1.close()
        inpFile2.close()
        intFile.close()
    move_lang(opt.src_lang)
    move_lang(opt.trg_lang)


def load_dev_dataset():
    """The function which loads the dev dataset"""
    opt = Opt.get_instance()
    opt.dev_dataset = f'data/{opt.src_lang}/DEV-{opt.src_lang}-{opt.trg_lang}.'

    with open(opt.dev_dataset + opt.src_lang, 'r', encoding='utf-8') as f:
        opt.dev_src_sentences = f.read().split('\n')[:2000]
    with open(opt.dev_dataset + opt.trg_lang, 'r', encoding='utf-8') as f:
        opt.dev_trg_sentences = f.read().split('\n')[:2000]


def train_spm_model():
    opt = Opt.get_instance()

    trainingOption = (f"--input={opt.input_file}{opt.src_lang} "
                      f"--model_prefix={opt.model_file}{opt.src_lang} "
                      f"--vocab_size=8000 --character_coverage=1.0 "
                      f"--model_type=BPE --pad_id=3 --bos_id=-1 --eos_id=-1 ")

    spm.SentencePieceTrainer.train(trainingOption)

    trainingOption = (f"--input={opt.input_file}{opt.src_lang} "
                      f"--model_prefix={opt.model_file}{opt.trg_lang} "
                      f"--vocab_size=8000 --character_coverage=1.0 "
                      f"--model_type=BPE --pad_id=3 --bos_id=-1 --eos_id=-1 ")

    spm.SentencePieceTrainer.train(trainingOption)


def create_models():
    """
    A function that loads the sentence piece model
    """
    opt = Opt.get_instance()

    print("initlizing sentence processors")
    opt.src_processor = spm.SentencePieceProcessor()
    opt.src_processor.Init(model_file=f'{opt.model_file}{opt.src_lang}.model')
    opt.trg_processor = spm.SentencePieceProcessor()
    opt.trg_processor.Init(model_file=f'{opt.model_file}{opt.trg_lang}.model')

    opt.src_pad = opt.src_processor.pad_id()
    opt.trg_pad = opt.trg_processor.pad_id()
    opt.trg_bos = opt.trg_processor.bos_id()
    opt.trg_eos = opt.trg_processor.eos_id()


def create_dataset():
    """
    A function to read-in the dataset, tokenize the sentences and place them into the appropriate bin.
    Finally, it saves the dataset to prevent the need to run this program again and again.
    """
    opt = Opt.get_instance()

    opt.bins = [i for i in range(10, opt.max_len + 1)]

    if opt.dataset is not None and os.path.exists(opt.dataset):
        print('loading saved dataset...')
        with open(opt.dataset, 'rb') as f:
            opt.src_bins = pickle.load(f)
            opt.trg_bins = pickle.load(f)

        print({s: len(opt.src_bins[s]) for s in opt.bins})
        return

    print('reading datasets')
    with open(opt.src_data_path, 'r', encoding='utf-8') as f:
        opt.src_data = f.read().split('\n')
    with open(opt.trg_data_path, 'r', encoding='utf-8') as f:
        opt.trg_data = f.read().split('\n')

    opt.src_bins = {i: [] for i in opt.bins}
    opt.trg_bins = {i: [] for i in opt.bins}

    print('tokenizing and bining...')
    for i in tnrange(len(opt.src_data)):
        src = opt.src_data[i]
        trg = opt.trg_data[i]
        # for i, (src, trg) in enumerate(zip(opt.src_data, opt.trg_data)):
        src = opt.src_processor.encode(src)
        trg = [opt.trg_bos] + opt.trg_processor.encode(trg) + [opt.trg_eos]
        opt.src_data[i] = 0
        opt.trg_data[i] = 0

        lsrc = len(src)
        ltrg = len(trg)
        if lsrc > opt.max_len or ltrg > opt.max_len:
            continue

        for v in opt.bins:
            if lsrc <= v and ltrg <= v:
                for _ in range(lsrc, v):
                    src.append(opt.src_pad)
                for _ in range(ltrg, v):
                    trg.append(opt.trg_pad)

                opt.src_bins[v].append(src)
                opt.trg_bins[v].append(trg)
                break

    if opt.dataset is not None:
        with open(opt.dataset, 'wb') as f:
            pickle.dump(opt.src_bins, f)
            pickle.dump(opt.trg_bins, f)

    temp = {s: len(opt.src_bins[s]) for s in opt.bins}
    opt.train_len = sum([temp[v] for v in opt.bins])
    print(temp)