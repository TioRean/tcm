from loadmod import TransformerLoader
from sql import SQLAdministrator
import os
import torch.nn as nn


class CMBrain:
    def __init__(self, transformer_path, sql_name='Gen', decode_method:str='top_k_greedy_decode',
                 search_width:int=3):
        self.transformer_model = TransformerLoader(transformer_path)
        self.sql = SQLAdministrator(sql_name)
        self.decode_method = decode_method
        self.search_width = search_width

    def apply_transformer(self, src:dict, top_k:int=1):
        return self.transformer_model.use_model([src['cls']], src['elems'], self.decode_method,
                                                   top_k, start_symbol=1)

class Util:
    @staticmethod
    def _read_src_from_txt(file_path:str):
        with open(file_path, 'r') as f:
            line = f.readline()
            while line:
                segments = line.split('\t')
                cls = segments[0]
                elems = segments[1].split(', ')
                yield {'cls':cls, 'elems':elems}
                line = f.readline()

    @staticmethod
    def cal_params(model: nn.Module):
        num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_non_learnable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return num_learnable_params, num_non_learnable_params

def map(inp:dict):
    path_checkpoint = os.path.join('ModelDict', 'checkpoint_best.pth')
    cmb = CMBrain(path_checkpoint, sql_name='Gen', decode_method='top_k_greedy_decode')
    inp['elems'] = inp['elems'].split(', ')
    result = cmb.apply_transformer(inp, 1)[0]
    print(result)
    return result


