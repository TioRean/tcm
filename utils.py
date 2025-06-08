import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import torchtext.transforms as T
import io
import math
from datapipe import StandardDataPipe


class CommonUtil:
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            CommonUtil.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    @staticmethod
    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    @staticmethod
    def get_std_opt(model, optimizer):
        return CommonUtil.NoamOpt(model.src_embed[0].d_model, 2, 4000, optimizer)

    class NoamOpt:
        def __init__(self, model_size, factor, warmup, optimizer):
            self.optimizer = optimizer
            self._step = 0
            self.warmup = warmup
            self.factor = factor
            self.model_size = model_size
            self._rate = 0

        def step(self):
            self._step += 1
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
            self.optimizer.step()

        def rate(self, step=None):
            if step is None:
                step = self._step
            return self.factor * \
                (self.model_size ** (-0.5) *
                 min(step ** (-0.5), step * self.warmup ** (-1.5)))

    class LabelSmoothing(nn.Module):
        def __init__(self, size, padding_idx=None, smoothing=0.1):
            super(CommonUtil.LabelSmoothing, self).__init__()
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.padding_idx = padding_idx
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.size = size
            self.true_dist = None

        def forward(self, x, target):
            assert x.size(1) == self.size
            true_dist = x.data.clone()
            true_dist.fill_(self.smoothing / (self.size - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if self.padding_idx is not None:
                true_dist[:, self.padding_idx] = 0
                mask = torch.nonzero(target.data == self.padding_idx)
                if mask.dim() > 0:
                    true_dist.index_fill_(0, mask.squeeze(), 0.0)
            self.true_dist = true_dist
            return self.criterion(x, Variable(true_dist, requires_grad=False))

    class SimpleLossCompute:
        def __init__(self, generator, criterion, opt):
            self.generator = generator
            self.criterion = criterion
            self.opt = opt
            self.training = True

        def __call__(self, x, y, norm):
            x = self.generator(x)
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
            if self.training:
                loss.backward()
                if self.opt is not None:
                    self.opt.step()
                    self.opt.optimizer.zero_grad()
            return loss.item() * norm

        def train(self, train: bool = True):
            self.training = train

        def eval(self):
            self.train(False)

    @staticmethod
    def save_model(mod, dir_path, current_epoch, model, stp_opt, label_smoothing, train_loss, val_loss, smallest_loss,
                   vocabs, emsize, nhid, nlayer, nhead, dropout, smoothing):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': stp_opt.optimizer.state_dict(),
            'opt_steps': stp_opt._step,
            'opt_lr': stp_opt._rate,
            'label_smoothing': label_smoothing.state_dict(),
            'vocabs': vocabs,
            'current_epoch': current_epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'smallest_loss': smallest_loss,
            'save_dir_path': dir_path,
            'emsize': emsize,
            'nhid': nhid,
            'nlayer': nlayer,
            'nhead': nhead,
            'dropout': dropout,
            'smoothing': smoothing
        }
        if mod == 'latest':
            torch.save(checkpoint, os.path.join(dir_path, 'checkpoint_latest.pth'))
            print('末次模型保存成功')
        elif mod == 'best':
            torch.save(checkpoint, os.path.join(dir_path, 'checkpoint_best.pth'))
            print('最佳模型保存成功')
        else:
            return ValueError('mod can only be latest or best')

    @staticmethod
    def transform_test_data(file_path, save_path):
        content = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                classification = line.split('\t')[0]
                elems = line.replace('\n', '').split('\t')[1].split('，')
                new_line = classification + '\t' + str(elems) + '\t\t\n'
                new_line = new_line.replace('[', '').replace(']', '').replace("'", '')
                content.append(new_line)
        with open(save_path, 'w') as f:
            for line in content:
                f.write(line)
            f.flush()


class TransformerUtil(CommonUtil):
    @staticmethod
    def data_generator(iterable):
        for data in iterable:
            segment_src = Variable(data[0], requires_grad=False)
            src = Variable(data[1], requires_grad=False)
            trg = Variable(data[2], requires_grad=False)
            yield TransformerUtil.Batch(segment_src, src, trg)

    class Batch:
        def __init__(self, segment_src, src, trg, pad=0):
            self.segment_src = segment_src
            self.src = src
            self.src_mask = (src != pad).unsqueeze(-2)
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = CommonUtil.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
            if torch.cuda.is_available():
                self.segment_src = self.segment_src.cuda()
                self.src = self.src.cuda()
                self.trg = self.trg.cuda()
                self.src_mask = self.src_mask.cuda()
                self.trg_mask = self.trg_mask.cuda()
                self.trg_y = self.trg_y.cuda()

    @staticmethod
    def greedy_decode(model, segment_src, src, src_mask, max_len, start_symbol):
        memory = model.encode(segment_src, src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = model.decode(memory, src_mask, Variable(ys),
                               Variable(CommonUtil.subsequent_mask(ys.size(1)).type_as(src.data)))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == 3:
                break
        return ys

    @staticmethod
    def top_k_greedy_decode(model, segment_src, src, src_mask, max_len=30, start_symbol=1, selected_k=1):
        memory, ys = TransformerUtil._top_k_greedy_decode(model, segment_src, src, src_mask, max_len, start_symbol, selected_k)
        return ys

    @staticmethod
    def _top_k_greedy_decode(model, segment_src, src, src_mask, max_len=30, start_symbol=1, selected_k=1):
        memory = model.encode(segment_src, src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len):
            out = model.decode(memory, src_mask, Variable(ys),
                               Variable(CommonUtil.subsequent_mask(ys.size(1)).type_as(src.data)))
            prob = model.generator(out[:, -1])
            if i == 0:
                _, next_words = torch.topk(prob, k=selected_k, dim=-1)
                next_word = next_words[0, selected_k - 1]
            else:
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == 3:
                break
        return memory, ys

    @staticmethod
    def idx_to_seq(indices, index_to_string):
        result = []
        for idx in indices:
            string = index_to_string[idx]
            if string == '<eos>' or string == '<sos>' or string == '<pad>':
                continue
            elif string in result:
                continue
            else:
                result.append(string)
        return str(result).replace('[', '').replace(']', '').replace("'", '')

    @staticmethod
    def count_accuracy(preds_tensor, trgs_tensor):
        preds_scalar = [set(pred.tolist()) for pred in preds_tensor]
        trgs_scaler = [set(trg.tolist()) for trg in trgs_tensor]
        same_element_num = 0
        trg_element_total_num = 0
        for pred, trg in zip(preds_scalar, trgs_scaler):
            pred = list(filter(lambda x: x != 0 and x != 1 and x != 2, pred))
            trg = list(filter(lambda x: x != 0 and x != 1 and x != 2, trg))
            same_element_num += len(set(pred) & set(trg))
            trg_element_total_num += len(trg)
        return same_element_num / trg_element_total_num

    @staticmethod
    def get_preds_tensor_list(decode_method, model, segment_srcs, srcs, src_masks, selected_k=1,
                              start_symbol=1, max_len=30):
        if decode_method == 'greedy_decode':
            preds_tensor = [(TransformerUtil.greedy_decode(model,
                                                           segment_src.unsqueeze(0),
                                                           src.unsqueeze(0),
                                                           mask.unsqueeze(0),
                                                           max_len,
                                                           start_symbol)).squeeze(0)
                            for segment_src, src, mask in zip(list(segment_srcs), list(srcs), list(src_masks))]
        elif decode_method == 'top_k_greedy_decode':
            preds_tensor = [(TransformerUtil.top_k_greedy_decode(model,
                                                                 segment_src.unsqueeze(0),
                                                                 src.unsqueeze(0),
                                                                 mask.unsqueeze(0),
                                                                 max_len,
                                                                 start_symbol,
                                                                 selected_k)).squeeze(0)
                            for segment_src, src, mask in zip(list(segment_srcs), list(srcs), list(src_masks))]
        else:
            raise Exception
        return preds_tensor

    @staticmethod
    def _getTransform(vocab, mod: int):
        text_transform = StandardDataPipe._getTransform(vocab, mod)
        return text_transform

    @staticmethod
    def test_model(data_iter, model, decode_method, vocabs, if_save=False, save_path=None,
                   top_k: int = 1, search_width: int = 3, start_symbol: int = 1, max_len: int = 30):
        model.eval()
        src_itos = vocabs[0].get_itos()
        trg_itos = vocabs[1].get_itos()
        print('==================START TESTING MODEL==================')
        for i, batch in enumerate(data_iter):
            srcs = torch.split(batch.src, 1, 0)
            srcs_tensor = [src.squeeze(0) for src in srcs]

            for selected_k in range(1, top_k + 1):
                preds_tensor = TransformerUtil.get_preds_tensor_list(decode_method,
                                                                     model,
                                                                     batch.src,
                                                                     batch.src_mask,
                                                                     selected_k,
                                                                     search_width,
                                                                     start_symbol,
                                                                     max_len)
                for src, pred in zip(srcs_tensor, preds_tensor):
                    src = TransformerUtil.idx_to_seq(src, src_itos)
                    pred = TransformerUtil.idx_to_seq(pred, trg_itos)
                    print(f'输入方证: {src}')
                    print(f'方药_TOP{selected_k}: {pred}')
                    if if_save is True:
                        with open(save_path, 'a') as f:
                            f.write(f'输入方证: {src} \n')
                            f.write(f'方药_TOP{selected_k}: {pred} \n')
                print('-----------------NEXT BATCH-----------------')

    @staticmethod
    def src_to_trg(model, decode_method, cls, elems, vocabs,
                   top_k: int = 1, start_symbol: int = 1, max_len: int = 30):
        target_itos = vocabs[1].get_itos()
        cls = TransformerUtil._getTransform(vocabs[0], mod=1)(cls)
        elems = TransformerUtil._getTransform(vocabs[0], mod=1)(elems)
        segment_src = [1] * len(cls) + [2] * len(elems)
        src = cls + elems
        src_tensor = torch.LongTensor([src])
        segment_src_tensor = torch.LongTensor([segment_src])
        source_mask = torch.ones(src_tensor.shape)
        if torch.cuda.is_available():
            src_tensor = src_tensor.cuda()
            segment_src_tensor = segment_src_tensor.cuda()
            source_mask = source_mask.cuda()
        model.eval()
        result = []
        for selected_k in range(1, top_k + 1):
            if decode_method == 'greedy_decode':
                ys = TransformerUtil.greedy_decode(model,
                                                   segment_src_tensor,
                                                   src_tensor,
                                                   source_mask,
                                                   max_len,
                                                   start_symbol)
            elif decode_method == 'top_k_greedy_decode':
                ys = TransformerUtil.top_k_greedy_decode(model,
                                                         segment_src_tensor,
                                                         src_tensor,
                                                         source_mask,
                                                         max_len,
                                                         start_symbol,
                                                         selected_k)
            else:
                raise Exception
            ys = ys.squeeze(0)
            ys = TransformerUtil.idx_to_seq(ys, target_itos)
            result.append(ys)
        return result
