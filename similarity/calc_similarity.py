import sentencepiece as spm
import os
from collections import defaultdict
from math import sqrt
import numpy as np
from tqdm import tqdm
from math import log2 as log
from translation.utils import Opt


def preprocessCountDictionary(counts):
    sum1 = sum(counts.values())
    output = np.array([counts[i] / sum1 for i in range(8000)])
    return output


def HellingerDistance(count1, count2):
    list1 = preprocessCountDictionary(count1)
    list2 = preprocessCountDictionary(count2)

    list1 = np.sqrt(list1)
    list2 = np.sqrt(list2)

    output = (list1 - list2)**2
    output = output.sum() / sqrt(2)

    return output


def KullbackLeibler(P, Q):
    output = 0
    for p, q in zip(P, Q):
        if p * q != 0:
            output += p * log(p / q)

    return output


def KullbackLeiblerDivergence(count1, count2):
    P = preprocessCountDictionary(count1)
    Q = preprocessCountDictionary(count2)

    return (KullbackLeibler(P, Q) + KullbackLeibler(Q, P)) / 2


def JensenShannonDistance(count1, count2):
    list1 = preprocessCountDictionary(count1)
    list2 = preprocessCountDictionary(count2)

    intermediateList = (list1 + list2) / 2

    return (KullbackLeibler(list1, intermediateList) + KullbackLeibler(list2, intermediateList)) / 2


def getCounts():
    opt = Opt.get_instance()
    os.chdir(opt.dir_name)

    concatenatedFile = f"ConcatenatedFile"
    if not os.path.exists(concatenatedFile):
        with open(concatenatedFile, 'w') as fout:
            with open(f'{opt.dir_name}.{opt.src_lang}') as fin:
                fout.write(fin.read() + '\n')

            with open(f'{opt.dir_name}.{opt.trg_lang}') as fin:
                fout.write(fin.read() + '\n')

    trainingOption = (f"--input={concatenatedFile} "
                      f"--model_prefix={concatenatedFile} "
                      f"--vocab_size=8000 --character_coverage=0.99 "
                      f"--model_type=bpe --pad_id=-1 --bos_id=-1 --eos_id=-1 ")

    if not os.path.exists(concatenatedFile + '.model'):
        spm.SentencePieceTrainer.train(trainingOption)

    processor = spm.SentencePieceProcessor()
    processor.Init(model_file=concatenatedFile + ".model")

    lang1Count = defaultdict(int)
    lang2Count = defaultdict(int)

    with open(f'{opt.dir_name}.{opt.src_lang}') as fin:
        for line in tqdm(fin.readlines()):
            line = processor.encode(line.strip('\n'))
            for char in line:
                lang1Count[char] += 1

    with open(f'{opt.dir_name}.{opt.trg_lang}') as fin:
        for line in tqdm(fin.readlines()):
            line = processor.encode(line.strip('\n'))
            for char in line:
                lang2Count[char] += 1

    os.chdir('..')

    return lang1Count, lang2Count


