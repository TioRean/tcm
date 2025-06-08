from utils import TransformerUtil
import torch
from model import TransformerModel

class ModelLoader:
    def __init__(self, checkpoint_path: str):
        # checkpoint_path：checkpoint路径
        self.checkpoint = torch.load(checkpoint_path)
        self.emsize = self.checkpoint['emsize']
        self.nhid = self.checkpoint['nhid']
        self.nlayer = self.checkpoint['nlayer']
        self.nhead = self.checkpoint['nhead']
        self.dropout = self.checkpoint['dropout']
        self.smoothing = self.checkpoint['smoothing']
        self.current_epoch = self.checkpoint['current_epoch']
        self.train_loss = self.checkpoint['train_loss']
        self.val_loss = self.checkpoint['val_loss']
        try:
            self.smallest_loss = self.checkpoint['smallest_loss']
        except Exception:
            self.smallest_loss = self.val_loss
        self.save_dir_path = self.checkpoint['save_dir_path']
        self.vocabs = self.checkpoint['vocabs']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.std_optimizer = None
        self.label_smoothing = None
        self.loss_compute = None

    def init_loss_fn_and_opt(self):
        pass

    def init_val_loss(self):
        self.val_loss = 100000000000

class TransformerLoader(ModelLoader):
    def __init__(self, checkpoint_path):
        '''
        用于读取加载Transformer模型的类
        :param checkpoint_path: Transformer模型参数的存储地址
        '''
        super(TransformerLoader, self).__init__(checkpoint_path)
        self.transformer_model_obj = TransformerModel(self.vocabs, self.save_dir_path,
                                                      emsize=self.emsize, nhid=self.nhid, nlayer=self.nlayer,
                                                      nhead=self.nhead, dropout=self.dropout, smoothing=self.smoothing)
        self.model = self.transformer_model_obj.model
        self.model.load_state_dict(self.checkpoint['model'])
        self.init_loss_fn_and_opt()

    def init_loss_fn_and_opt(self):
        assert self.model is not None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-9)
        self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        self.std_optimizer = TransformerUtil.get_std_opt(self.model, self.optimizer)
        self.std_optimizer._step = self.checkpoint['opt_steps']
        self.std_optimizer._rate = self.checkpoint['opt_lr']
        self.label_smoothing = TransformerUtil.LabelSmoothing(size=len(self.vocabs[1]), padding_idx=0,
                                                              smoothing=self.smoothing)
        self.label_smoothing.load_state_dict(self.checkpoint['label_smoothing'])
        self.loss_compute = TransformerUtil.SimpleLossCompute(self.model.generator, self.label_smoothing,
                                                              self.std_optimizer)

    def train_model(self, train_dp_gen, val_dp_gen, del_used_dataset=True, total_epochs=100,
                    print_interval=500, print_instances=True, save_dir_path: str = None) -> None:
        if save_dir_path is not None:
            self.save_dir_path = save_dir_path
        TransformerModel.train_model_(self.emsize, self.nhid, self.nlayer, self.nhead, self.dropout, self.smoothing,
                                      self.model, self.loss_compute, self.std_optimizer, self.label_smoothing,
                                      train_dp_gen, val_dp_gen, self.vocabs, self.save_dir_path, total_epochs,
                                      self.current_epoch, self.smallest_loss, del_used_dataset, print_interval,
                                      print_instances)

    def test_model(self, dp_gen, decode_method:str = 'top_k_greedy_decode', if_save: bool = False, save_path: str = None,
                   top_k:int=1, search_width:int=3, start_symbol:int=1, max_len:int=30):
        _, test_dp = next(dp_gen)
        TransformerUtil.test_model(TransformerUtil.data_generator(test_dp), self.model, decode_method,
                                   self.vocabs, if_save, save_path, top_k, search_width, start_symbol,
                                   max_len)


    def use_model(self, src_head, src, decode_method='top_k_greedy_decode', top_k=1, start_symbol=1, max_len=30) -> str:
        result = TransformerUtil.src_to_trg(self.model, decode_method, src_head, src, self.vocabs, top_k, start_symbol, max_len)
        return result

    def use_model_and_get_intermedia_result(self, src_head, src, decode_method='top_k_greedy_decode', top_k=1,
                  start_symbol=1, max_len=30) -> str:
        result = TransformerUtil.src_to_trg(self.model, decode_method, src_head, src, self.vocabs, top_k, start_symbol, max_len)
        return result
