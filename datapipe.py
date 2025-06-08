import torchdata.datapipes as dp
import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator
import os

class Datapipe:
    @staticmethod
    def _make_train_val_datapipe(datapipe_cls, train_paths, val_paths, vocabs=None, train_bs=20, val_bs=10,
                                 batch_num=5):
        if vocabs is not None:
            train_datapipe = datapipe_cls(train_paths, vocabs=vocabs, batch_size=train_bs, batch_num=batch_num)
        else:
            train_datapipe = datapipe_cls(train_paths, batch_size=train_bs, batch_num=batch_num)
            vocabs = train_datapipe.vocabs
        train_dp_iter = train_datapipe.get_datapipe()
        val_datapipe = datapipe_cls(val_paths, vocabs=vocabs, batch_size=val_bs, batch_num=batch_num)
        val_dp_iter = val_datapipe.get_datapipe()
        return train_dp_iter, val_dp_iter, vocabs

    @staticmethod
    def make_standard_train_val_datapipe(train_paths:list, val_paths:list, vocabs=None, train_bs=20, val_bs=10,
                                         batch_num=5):
        train_dp_iter, val_dp_iter, vocabs = \
            Datapipe._make_train_val_datapipe(StandardDataPipe, train_paths, val_paths, vocabs, train_bs, val_bs,
                                              batch_num)
        return train_dp_iter, val_dp_iter, vocabs

    @staticmethod
    def _make_datapipe(datapipe_cls, path, vocabs, batch_size=20, batch_num=5):
        if type(path) is str:
            path = [path]
        if vocabs is not None:
            datapipe = datapipe_cls(path, vocabs=vocabs, batch_size=batch_size, batch_num=batch_num)
        else:
            datapipe = datapipe_cls(path, batch_size=batch_size, batch_num=batch_num)
            vocabs = datapipe.vocabs
        dp_iter = datapipe.get_datapipe()
        return dp_iter, vocabs

    @staticmethod
    def make_standard_datapipe(paths: list or str, vocabs=None, batch_size=20, batch_num=5):
        dp_iter, vocabs = Datapipe._make_datapipe(StandardDataPipe, paths, vocabs, batch_size, batch_num)
        return dp_iter, vocabs

class StandardDataPipe:
    def __init__(self, FILE_PATH_LIST, vocabs=None, batch_size=20, batch_num=5, bucket_num=1):
        self.FILE_PATH_LIST = FILE_PATH_LIST
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.bucket_num = bucket_num
        if vocabs is not None:
            self.vocabs = vocabs
            self.source_vocab, self.target_vocab = vocabs
        else:
             self.vocabs = self.get_vocab()
             self.source_vocab, self.target_vocab = self.vocabs

    def get_datapipe(self):
        for FILE_PATH in self.FILE_PATH_LIST:
            if not os.path.exists(FILE_PATH):
                continue
            try:
                data_pipe = dp.iter.IterableWrapper([FILE_PATH])
                data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
            except FileNotFoundError: continue
            else:
                data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)
                data_pipe = data_pipe.map(self._applyTransform)
                data_pipe = data_pipe.bucketbatch(
                    batch_size=self.batch_size, batch_num=self.batch_num, bucket_num=self.bucket_num,
                    use_in_batch_shuffle=True, sort_key=self._sort_bucket)
                data_pipe = data_pipe.map(self._separateSourceTarget)
                data_pipe = data_pipe.map(self._applyPadding)
                yield FILE_PATH, data_pipe


    def get_vocab(self):
        for FILE_PATH in self.FILE_PATH_LIST:
            if not os.path.exists(FILE_PATH):
                continue
            try:
                data_pipe = dp.iter.IterableWrapper([FILE_PATH])
                data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
            except FileNotFoundError: continue
            else:
                data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)
                source_vocab = build_vocab_from_iterator(
                    self._getTokens(data_pipe, 0),
                    min_freq=1,
                    specials=['<pad>', '<sep>', '<unk>'],
                    special_first=True
                )
                source_vocab.set_default_index(source_vocab['<unk>'])

                # target（药物）的词表
                target_vocab = build_vocab_from_iterator(
                    self._getTokens(data_pipe, 1),
                    min_freq=1,
                    specials=['<pad>', '<sos>', '<sep>', '<eos>', '<unk>'],
                    special_first=True
                )
                target_vocab.set_default_index(target_vocab['<unk>'])
                return source_vocab, target_vocab

    @staticmethod
    def _getTransform(vocab, mod:int):
        if mod == 1:
            text_transform = T.Sequential(
                T.VocabTransform(vocab=vocab))
        elif mod == 2:
            text_transform = T.Sequential(
                T.VocabTransform(vocab=vocab),
                T.AddToken(1, begin=True),
                T.AddToken(2, begin=False))
        elif mod == 3:
            text_transform = T.Sequential(
                T.VocabTransform(vocab=vocab),
                T.AddToken(3, begin=False))
        else:
            return ValueError('mod can only be 0 or 1 or 2')
        return text_transform

    def _applyTransform(self, sequence_group):
        cls = self._getTransform(self.source_vocab, mod=1)(self._tokenize(sequence_group[0]))
        elems = self._getTransform(self.source_vocab, mod=1)(self._tokenize(sequence_group[1]))
        segment_src = [1] * len(cls) + [2] * len(elems)
        token_src = cls + elems

        prscrpt = self._getTransform(self.target_vocab, mod=2)(self._tokenize(sequence_group[2]))
        medic = self._getTransform(self.target_vocab, mod=3)(self._tokenize(sequence_group[3]))
        token_trg = prscrpt + medic
        return (segment_src, token_src, token_trg)

    @staticmethod
    def _tokenize(text):
        return text.split(', ')

    def _getTokens(self, data_iter, place):
        for cls, elems, prscrpt, medic in data_iter:
            if place == 0:
                yield self._tokenize(elems) + self._tokenize(cls)
            else:
                yield self._tokenize(prscrpt) + self._tokenize(medic)

    def _sort_bucket(self, bucket):
        return sorted(bucket, key=lambda x: (len(x[0]), len(x[-1])))

    def _separateSourceTarget(self, sequence_pairs):
        segment_sources, token_sources, token_targets = zip(*sequence_pairs)
        return segment_sources, token_sources, token_targets

    def _applyPadding(self, pair_of_sequences):
        return (T.ToTensor(0)(list(pair_of_sequences[0])), T.ToTensor(0)(list(pair_of_sequences[1])),
                T.ToTensor(0)(list(pair_of_sequences[2])))

