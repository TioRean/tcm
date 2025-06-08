import torch.nn.functional as F
from torch.autograd import Variable
import math
from utils import TransformerUtil
import torch
from torch import nn
import os
import time
import copy

class Transformer:
    def __init__(self, emsize: int = 256, nhid: int = 512, nlayer: int = 4, nhead: int = 4,
                 dropout: float = 0.1, smoothing: float = 0.1):
        self.emsize = emsize
        self.nhid = nhid
        self.nlayer = nlayer
        self.nhead = nhead
        self.dropout = dropout
        self.smoothing = smoothing

    class Embeddings(nn.Module):
        def __init__(self, d_model, vocab):
            super(Transformer.Embeddings, self).__init__()
            self.lut = nn.Embedding(vocab, d_model)
            self.d_model = d_model

        def forward(self, x):
            return self.lut(x) * math.sqrt(self.d_model)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout, max_len=5000):
            super(Transformer.PositionalEncoding, self).__init__()

            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)

            position = torch.arange(0, max_len, ).unsqueeze(1)

            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)

            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
            return self.dropout(x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        def _softmax_1(x, dim=-1):
            row_max = x.max(dim=dim)[0]
            row_max = row_max.unsqueeze(-1)
            x = torch.sub(x, row_max)
            x_exp = torch.exp(x)
            x_sum = torch.sum(x_exp, dim=dim, keepdims=True)
            res = torch.div(x_exp, 1 + x_sum)
            return res

        p_attn = _softmax_1(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    @staticmethod
    def _clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    class MultiHeadedAttendtion(nn.Module):
        def __init__(self, head, embedding_dim, dropout=0.1):
            super(Transformer.MultiHeadedAttendtion, self).__init__()

            assert embedding_dim % head == 0
            self.d_k = embedding_dim // head
            self.head = head
            self.embedding_dim = embedding_dim
            self.linears = Transformer._clones(nn.Linear(embedding_dim, embedding_dim), 4)

            self.attn = None

            self.dropout = nn.Dropout(p=dropout)

        def forward(self, query, key, value, mask=None):
            if mask is not None:
                mask = mask.unsqueeze(1)
            batch_size = query.size(0)
            query, key, value = \
                [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                 for model, x in zip(self.linears, (query, key, value))]

            x, self.attn = Transformer.attention(query, key, value, mask=mask, dropout=self.dropout)


            x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

            return self.linears[-1](x)

    class PositionwiseFeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super(Transformer.PositionwiseFeedForward, self).__init__()
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            return self.w2(self.dropout(F.silu(self.w1(x))))

    class LayerNorm(nn.Module):
        def __init__(self, feature, eps=1e-6):
            super(Transformer.LayerNorm, self).__init__()

            self.a2 = nn.Parameter(torch.ones(feature))
            self.b2 = nn.Parameter(torch.zeros(feature))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.a2 * (x - mean) / (std + self.eps) + self.b2

    class SublayerConnection(nn.Module):
        def __init__(self, size, dropout=0.1):
            super(Transformer.SublayerConnection, self).__init__()
            self.norm = Transformer.LayerNorm(size)
            self.dropout = nn.Dropout(p=dropout)
            self.size = size

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

    class EncodeLayer(nn.Module):
        def __init__(self, size, self_attn, feed_forward, dropout):
            super(Transformer.EncodeLayer, self).__init__()

            self.self_attn = self_attn
            self.feed_forward = feed_forward
            self.size = size

            self.sublayer = Transformer._clones(Transformer.SublayerConnection(size, dropout), 2)

        def forward(self, x, mask):
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)

    class Encoder(nn.Module):
        def __init__(self, layer, N):
            super(Transformer.Encoder, self).__init__()
            self.layers = Transformer._clones(layer, N)
            self.norm = Transformer.LayerNorm(layer.size)

        def forward(self, x, mask):
            for layer in self.layers:
                x = layer(x, mask)
            return self.norm(x)

    class DecoderLayer(nn.Module):
        def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
            super(Transformer.DecoderLayer, self).__init__()

            self.size = size
            self.self_attn = self_attn
            self.src_attn = src_attn
            self.feed_forward = feed_forward
            self.dropout = dropout

            self.sublayer = Transformer._clones(Transformer.SublayerConnection(size, dropout), 3)

        def forward(self, x, memory, source_mask, target_mask):

            m = memory
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
            return self.sublayer[2](x, self.feed_forward)

    class Decoder(nn.Module):
        def __init__(self, layer, N):
            super(Transformer.Decoder, self).__init__()
            self.layers = Transformer._clones(layer, N)
            self.norm = Transformer.LayerNorm(layer.size)

        def forward(self, x, memory, source_mask, target_mask):
            for layer in self.layers:
                x = layer(x, memory, source_mask, target_mask)
            return self.norm(x)

    class Generator(nn.Module):
        def __init__(self, d_model, vocab_size):
            super(Transformer.Generator, self).__init__()
            self.project = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            return F.log_softmax(self.project(x), dim=-1)

    class EncoderDecoder(nn.Module):
        def __init__(self, encoder, decoder, segment_src_embed, source_embed, trg_embed, generator):
            super(Transformer.EncoderDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.segment_src_embed = segment_src_embed
            self.src_embed = source_embed
            self.trg_embed = trg_embed
            self.generator = generator

        def forward(self, segment_src, source, target, source_mask, target_mask):
            return self.decode(self.encode(segment_src, source, source_mask), source_mask,
                               target, target_mask)

        def encode(self, segment_src, source, source_mask):
            return self.encoder(self.src_embed(source) + self.segment_src_embed(segment_src), source_mask)

        def decode(self, memory, source_mask, target, target_mask):
            return self.decoder(self.trg_embed(target), memory, source_mask, target_mask)


class TransformerModel(Transformer):
    def __init__(self, vocabs: list or set, save_dir_path: str, emsize: int = 256,
                 nhid: int = 512, nlayer: int = 6, nhead: int = 4, dropout: float = 0.1, smoothing: float = 0.1):
        super(TransformerModel, self).__init__(emsize, nhid, nlayer, nhead, dropout, smoothing)
        self.vocabs = vocabs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir_path = save_dir_path
        self.src_vocab, self.trg_vocab = self.vocabs
        self.model = self._make_model(source_vocab=len(self.src_vocab), target_vocab=len(self.trg_vocab), N=nlayer,
                                      d_ff=nhid, head=nhead, dropout=dropout)
        self.model.to(self.device)
        self.optimizer = None
        self.std_optimizer = None
        self.label_smoothing = None
        self.loss_compute = None

    def init_loss_fn_and_opt(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9)
        self.std_optimizer = \
            TransformerUtil.get_std_opt(self.model, self.optimizer)

        self.label_smoothing = TransformerUtil.LabelSmoothing(size=len(self.trg_vocab), padding_idx=0,
                                                              smoothing=self.smoothing)
        self.label_smoothing.to(self.device)

        self.loss_compute = TransformerUtil.SimpleLossCompute(self.model.generator, self.label_smoothing,
                                                              self.std_optimizer)

    def print_config(self):

        print(f'save_dict_path: {self.save_dir_path}')
        print(f'model：transformer || loss_fn:KLDivLoss || optim:Adam ')
        print(f'emsize: {self.emsize}       || nhid：{self.nhid}         || nlayer: {self.nlayer}')
        print(f'nhead: {self.nhead}          || dropout: {self.dropout}       ||smoothing: {self.smoothing}')
        print(f'device: {self.device}')
        print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

    def train_model(self, train_dp_iter: iter, val_dp_iter: iter, total_epochs: int = 50, epoch_begin: int = 0,
                    del_used_dataset: bool = True, print_interval: int = 500, print_instances: bool = True):

        TransformerModel.train_model_(self.emsize, self.nhid, self.nlayer, self.nhead, self.dropout, self.smoothing,
                                      self.model, self.loss_compute, self.std_optimizer,
                                      self.label_smoothing, train_dp_iter, val_dp_iter, self.vocabs,
                                      self.save_dir_path, total_epochs, epoch_begin, None,
                                      del_used_dataset, print_interval, print_instances)

    def _make_model(self, source_vocab, target_vocab, N=6, d_model=256, d_ff=512, head=4, dropout=0.1):

        c = copy.deepcopy

        attn = self.MultiHeadedAttendtion(head, d_model)

        ff = self.PositionwiseFeedForward(d_model, d_ff, dropout)
        position = self.PositionalEncoding(d_model, dropout)
        model = self.EncoderDecoder(
            self.Encoder(self.EncodeLayer(d_model, c(attn), c(ff), dropout), N),
            self.Decoder(self.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(self.Embeddings(d_model, 3)),
            nn.Sequential(self.Embeddings(d_model, source_vocab)),
            nn.Sequential(self.Embeddings(d_model, target_vocab), c(position)),
            self.Generator(d_model, target_vocab))

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    @classmethod
    def _run_epoch(cls, data_iter, model, loss_compute, vocabs, src_itos, trg_itos, print_interval=2000,
                   print_instances=False):
        time_start = time.time()
        ntokens, total_tokens, total_loss = 0, 0, 0
        for i, batch in enumerate(data_iter):
            out = model.forward(batch.segment_src, batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            ntokens += batch.ntokens
            total_loss += loss
            total_tokens += batch.ntokens
            if i % print_interval == 1 and i != 1:
                elapsed = time.time() - time_start
                srcs = torch.split(batch.src, 1, 0)
                srcs_tensor = [src.squeeze(0) for src in srcs]
                trgs = torch.split(batch.trg, 1, 0)
                trgs_tensor = [trg.squeeze(0) for trg in trgs]
                preds_tensor = TransformerUtil.get_preds_tensor_list('greedy_decode',
                                                                     model,
                                                                     batch.segment_src,
                                                                     batch.src,
                                                                     batch.src_mask,
                                                                     batch.trg)
                accuracy = TransformerUtil.count_accuracy(preds_tensor, trgs_tensor)
                if print_instances and vocabs is not None:
                    for src, trg, pred in zip(srcs_tensor, trgs_tensor, preds_tensor):
                        print(f'src: {TransformerUtil.idx_to_seq(src, src_itos)}')
                        print(f'trg: {TransformerUtil.idx_to_seq(trg, trg_itos)}')
                        print(f'pred: {TransformerUtil.idx_to_seq(pred, trg_itos)}')
                        print('-----------------------------------------------------------')
                    print('==========================================================================================')
                print("Epoch Step: %d || Loss: %f || Accuracy: %f || Tokens per Sec: %f" %
                      (i, loss / batch.ntokens, accuracy, ntokens / elapsed))
                print('==========================================================================================')
                time_start = time.time()
                ntokens = 0
        return total_loss / total_tokens

    @classmethod
    def train_model_(cls, emsize:int, nhid:int, nlayer:int, nhead:int, dropout:float, smoothing:float,
                     model, loss_compute, stp_opt, label_smoothing, train_dp_iter, val_dp_iter, vocabs,
                     save_dir_path: str, total_epochs: int = 5, epoch_begin: int = 0,
                     last_loss=None, del_used_dataset: bool = True, print_interval: int = 500,
                     print_instances: bool = True) -> None:
        assert epoch_begin >= 0 and epoch_begin < total_epochs
        smallest_val_loss = last_loss if last_loss is not None else 1000000
        src_itos = vocabs[0].get_itos()
        trg_itos = vocabs[1].get_itos()

        for epoch in range(epoch_begin, total_epochs):
            print(f'---------------------当前第 {epoch}/{total_epochs} epoch--------------------')
            start = time.perf_counter()

            train_path, train_dp, = next(train_dp_iter)
            val_path, val_dp = next(val_dp_iter)

            print('当前为训练模式：')
            model.train()
            loss_compute.train()
            train_loss = cls._run_epoch(TransformerUtil.data_generator(train_dp), model, loss_compute, vocabs,
                                         src_itos, trg_itos, print_interval=print_interval)
            print('\n当前为验证模式: ')
            model.eval()
            loss_compute.eval()
            with torch.no_grad():
                val_loss = cls._run_epoch(TransformerUtil.data_generator(val_dp), model, loss_compute, vocabs,
                                           src_itos, trg_itos, print_interval=print_interval,
                                           print_instances=print_instances)
            print('==========================================================================================')
            print(f'本轮epoch结果: train_loss: {train_loss}, val_loss: {val_loss}')
            print(f'本轮epoch用时：{format((time.perf_counter() - start) / 3600, ".2f")}小时')
            print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

            if del_used_dataset:
                try:
                    os.remove(train_path)
                    os.remove(val_path)
                except FileNotFoundError:
                    pass

            if save_dir_path is not None:
                if val_loss < smallest_val_loss:
                    smallest_val_loss = val_loss
                    TransformerUtil.save_model('best', save_dir_path, epoch + 1, model, stp_opt, label_smoothing,
                                               train_loss, val_loss, smallest_val_loss, vocabs,
                                               emsize, nhid, nlayer, nhead, dropout, smoothing)

                TransformerUtil.save_model('latest', save_dir_path, epoch + 1, model, stp_opt, label_smoothing,
                                           train_loss, val_loss, smallest_val_loss, vocabs,
                                           emsize, nhid, nlayer, nhead, dropout, smoothing)
                print('\n')
