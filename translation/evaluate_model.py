from translation.utils import Opt, nopeak_mask

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import sacrebleu
import math
import re
from tqdm.notebook import tnrange


def k_best_outputs(outputs, pred, log_probs, i, k):
    """The function that picks the top k output (part of beam search)"""
    probs, idx = pred[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_probs.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = idx[row, col]

    log_probs = k_probs.unsqueeze(0)

    return outputs, log_probs


def beam_search(src, model):
    """The function that implements the beam search (pruned breadth-first search)"""
    opt = Opt.get_instance()
    bos_token = opt.trg_bos
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    encoder_output = model.encoder(src, src_mask)

    outputs = torch.LongTensor([[bos_token]]).to(opt.device)

    trg_mask = nopeak_mask(1)

    pred = model.linear(model.decoder(outputs, encoder_output, src_mask, trg_mask))
    pred = F.softmax(pred, dim=-1)

    probs, idx = pred[:, -1].data.topk(opt.k)
    log_probs = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.k, opt.max_len).long().to(opt.device)
    outputs[:, 0] = bos_token
    outputs[:, 1] = idx[0]

    encoder_outputs = torch.zeros(opt.k, encoder_output.size(-2), encoder_output.size(-1)).to(opt.device)
    encoder_outputs[:, :] = encoder_output[0]

    eos_token = opt.trg_eos
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
        trg_mask = nopeak_mask(i)
        pred = model.linear(model.decoder(outputs[:, :i],
                                          encoder_outputs, src_mask, trg_mask))
        pred = F.softmax(pred, dim=-1)
        outputs, log_probs = k_best_outputs(outputs, pred, log_probs, i, opt.k)
        ones = torch.nonzero(outputs == eos_token)
        # ones = (outputs==eos_token).nonzero() # Occurrences of end symbols for all input sentences.
        sequence_lengths = torch.zeros(len(outputs)).to(opt.device)
        for vec in ones:
            i = vec[0]
            if sequence_lengths[i] == 0:  # First end symbol has not been found yet
                sequence_lengths[i] = vec[1]  # Position of first end symbol

        complete_sentence_count = len([s for s in sequence_lengths if s > 0])

        if complete_sentence_count == opt.k:
            alpha = 0.7
            div = 1 / (sequence_lengths.type_as(log_probs) ** alpha)
            _, ind = torch.max(log_probs * div, 1)
            ind = ind.data[0]
            break

    if ind is None:
        length = (outputs[0] == eos_token).nonzero()[0]
        sentence_list = (outputs[0][1:length]).tolist()
        return ''.join(opt.trg_processor.decode(sentence_list)).replace('_', " ")

    else:
        length = (outputs[ind] == eos_token).nonzero()[0]
        sentence_list = (outputs[0][1:length]).tolist()
        return ''.join(opt.trg_processor.decode(sentence_list)).replace('_', " ")


def multiple_replace(dict, text):
    """A function that uses regex to replace certain parts of the sentence to make it suitable for printing"""
    # compiling the regex based on dictionary
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match i.e. x look up the value in the dictionary to replace
    return regex.sub(lambda x: dict[x.string[x.start():x.end()]], text)


def translate_sentence(sentence, model):
    """The function that uses beam search to translate the sentences"""
    opt = Opt.get_instance()
    model.eval()
    sentence = Variable(torch.LongTensor([opt.src_processor.encode(sentence)])).to(opt.device)

    sentence = beam_search(sentence, model)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)


def evaluate():
    """A function that evaluates the model using the dev set during the training step"""
    opt = Opt.get_instance()
    model = opt.model
    model.eval()

    refs = []
    hyp = []
    opt.skip = []

    tk2 = tnrange(len(opt.dev_src_sentences))
    for i in tk2:
        sentence = opt.dev_src_sentences[i]
        try:
            translated = translate_sentence(sentence.lower(), model)
        except:
            opt.skip.append(i)
        else:
            refs.append(opt.dev_trg_sentences[i])
            hyp.append(translated)
            if i < 10:
                hyp_print = f"hyp: {hyp[-1]}"
                ref_print = f"ref: {refs[-1]}"
                opt.log.print(hyp_print, shell=False)
                opt.log.print(ref_print, shell=False)
                opt.log.print('', shell=False)
                opt.log.flush()

        if i == len(opt.dev_src_sentences) - 1:
            opt.hyp = hyp
            opt.refs = refs

            bleu = sacrebleu.corpus_bleu(opt.hyp[:-1], [opt.refs[:-1]])
            ter = sacrebleu.corpus_ter(opt.hyp[:-1], [opt.refs[:-1]])
            tk2.set_postfix_str(f"({ter} || {bleu})")
            opt.log.print(f'{ter} || {bleu}', shell=False)

            return bleu, ter