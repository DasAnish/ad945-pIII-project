import torch
import torch.nn.functional as F
import torch.nn as nn
import sacrebleu
import numpy as np
import time
from utils import Opt, batch, create_masks
from tqdm.notebook import tnrange
import IPython
import matplotlib.pyplot as plt
from evaluate_model import evaluate
Tensor = torch.tensor


def train_epoch():
    opt = Opt.get_instance()
    model = opt.model
    model.train()
    optim = opt.optim

    start = time.time()
    temp = time.time()
    total_loss = 0

    tk1 = tnrange(opt.train_len)
    batch_gen = batch()

    last_loss = 0
    loss_at_save = 0
    r = opt.save_every / opt.print_every
    losses = []
    losses_at_save = []
    for i, (src_lis, trg_lis) in zip(tk1, batch_gen):
        optim.zero_grad()

        try:
            src_tensor = torch.LongTensor(src_lis).to(opt.device)
            src_tensor.requires_grad = False
            trg_np = np.array(trg_lis)
            trg_tensor = torch.LongTensor(trg_np[:, :-1]).to(opt.device)
            trg_tensor.requires_grad = False
        except:
            del src_tensor
            continue

        src_mask, trg_mask = create_masks(src_tensor, trg_tensor)

        preds = model(src_tensor, trg_tensor, src_mask, trg_mask)
        target = torch.LongTensor(trg_np[:, 1:]).to(opt.device).contiguous().view(-1)
        preds = preds.view(-1, preds.size(-1))
        loss = F.cross_entropy(preds, target, ignore_index=opt.trg_pad)

        loss.backward()
        total_loss += loss.item()
        opt.optim.step()
        opt.step += 1
        opt.optim.param_groups[0]['lr'] = (opt.model_dim ** (-0.5)) * \
                                          min(opt.step ** (-0.5),
                                              opt.step * (opt.warmup_steps ** (-1.5)))

        del src_mask, src_tensor, trg_mask, trg_tensor, preds, loss
        torch.cuda.empty_cache()

        if (opt.step+1) % opt.print_every == 1:
            IPython.display.clear_output(wait=True)
            diff = total_loss - last_loss
            diff = '%.3f' % diff
            last_loss = total_loss

            avg = "%.3f" % (total_loss / (opt.step - opt.starting_index))
            t = "%.3f" % (time.time() - temp)
            tt = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))

            output = f"time: {t}s, total: {tt}, loss = {avg}, step = {opt.step}, diff = {diff}"
            opt.log.print(output, shell=True)
            # tk1.set_postfix_str(", " + output + '\n')
            losses.append(diff)
            temp = time.time()

            plt.figure(figsize=(8, 6))
            plt.plot(losses)
            plt.show()

        if (opt.step+1) % opt.save_every == 0:
            model_name = f'{opt.path}/{opt.model_prefix}{opt.step}'
            opt.save_model.model_state_dict = model.state_dict()
            opt.save_model.optim_state_dict = opt.optim.state_dict()
            opt.save_model.save(model_name)

            avg = ((total_loss - loss_at_save) / r)
            loss_at_save = total_loss
            diff = avg - last_loss
            last_loss = avg

            avg = '%.3f' % avg
            diff = '%.3f' % diff
            output = f"Saving model: {model_name} | avg_loss: {avg} | diff: {diff}"
            opt.log.print(output, shell=True)
            # tk0.set_postfix_str(',' + output + '\n')
            losses_at_save.append(avg)
            plt.plot(losses_at_save, label='save_loss')
            plt.legend()
            plt.show()

            opt.log.flush()

    return total_loss


def train_model():
    opt = Opt.get_instance()
    losses = []
    while True:
        loss = train_epoch()
        losses.append(loss)
        plt.plot(losses)
        plt.title("epoch loss")
        plt.show()

        if len(losses) > 5:
            l1 = losses[-5:]
            l2 = losses[-6: -1]
            if sum(l1) - sum(l2) > 0.5:
                break








