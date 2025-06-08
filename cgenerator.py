import math
import random
import re
from itertools import chain, islice, tee
import copy
from random import shuffle
import os
from sql import SQLAdministrator
from dataclasses import dataclass, field
import numpy as np
from typing import Sequence, Union
from logger import get_logger


logger = get_logger('cgenerate')

class Util:
    @staticmethod
    def _in_range(num, start, end):
        return start <= num < end

    @staticmethod
    def _make_dir(*paths):
        for path in paths:
            if os.path.exists(path):
                continue
            os.makedirs(path, exist_ok=False)

    @staticmethod
    def _get_hierarchical_accessory(accessory: dict):
        content = dict()
        def get_hierarchical_dict(k, accessory):
            if k not in content.keys():
                content[k] = dict()
            for v in accessory[k]:
                if v not in content.keys():
                    content[v] = dict()
                content[v][k] = copy.deepcopy(content[k])
                if v in accessory.keys():
                    get_hierarchical_dict(v, accessory)
        def remove_duplicate(content: dict):
            temp = copy.deepcopy(content)
            for k, v in content.items():
                for x, y in content.items():
                    if k in y.keys():
                        if k in temp.keys():
                            temp.pop(k)
                if k in temp.keys():
                    temp[k] = remove_duplicate(temp[k])
            return temp
        for k in accessory.keys():
            get_hierarchical_dict(k, accessory)
        content = remove_duplicate(content)
        return content

    @staticmethod
    def _get_necessary_root_elem_list(root_elems_list):
        result = []
        for psc_num, rule_num, cls, cpl, opt, tong_pul, psc, med in root_elems_list:
            necessary_cpl = list(filter(lambda x: len(x) > 0 and 'null' not in x, cpl))
            necessary_opt = list(filter(lambda x: len(x) > 0 and 'null' not in x, opt[:-1]))
            for i, xs in enumerate(necessary_opt):
                xs = list(chain.from_iterable(x.split('&') for x in xs))
                xs = list(chain.from_iterable(x.split('/') for x in xs))
                necessary_opt[i] = xs
            processed_tp = []
            temp = dict()
            necessary_combi_dict = dict()
            for trait_li in tong_pul:
                for trait in trait_li:
                    trait_split = trait.replace('; ', '；').replace(';', '；').split('；')
                    if re.search('[Oo][A-Za-z]+', trait_split[1]):
                        if trait_split[1] not in temp.keys():
                            temp[trait_split[1]] = []
                        temp[trait_split[1]].append(trait_split[0])
                    if trait_split[2] != '':
                        cp_nes = list(filter(lambda x: len(x) > 0, trait_split[2].split('&')))
                        cp_nes = list(chain.from_iterable([xs.split('/') for xs in cp_nes]))
                        necessary_combi_dict[trait_split[0]] = cp_nes

            processed_tp.extend(temp.values())
            tong_pul = [[trait.replace('; ', '；').split('；')[0] for trait in trait_li]
                        for trait_li in tong_pul]
            processed_tp.extend(filter(lambda x: len(x) > 0 and 'null' not in x, tong_pul))
            result.append([psc_num, rule_num, cls, necessary_cpl, necessary_opt, processed_tp,
                           necessary_combi_dict, psc, med])
        return result

    # ----------------删除矛盾医案的工具---------------
    @staticmethod
    def _is_contradictory_inst(inst: Union[tuple, list, set], contradictory_elements_dict: dict):
        for elem in inst:
            if elem in contradictory_elements_dict.keys():
                contradictory_elems = contradictory_elements_dict[elem]
                if bool(set(inst) & set(contradictory_elems)):
                    return True
        return False

    # ----------------数据保存工具--------------------
    @staticmethod
    def _save_cases(save_path: str, cases: list):
        with open(save_path, 'w') as f:
            for case in cases:
                f.write(case)
            f.flush()

    @staticmethod
    def _save_to_one_dataset(save_path: str, gen_list: list, total_num: int, batch_size=30000000):
        if total_num <= batch_size:
            container = []
            for _, gen in gen_list:
                container.extend([case for case in gen])
            random.shuffle(container)
            Util._save_cases(save_path, container)
        else:
            temp_paths, read_files = Util._temp_save_to_disk(gen_list, save_path, total_num, batch_size)
            with open(save_path, 'w') as f:
                while len(read_files) > 0:
                    read_file = random.choice(read_files)
                    line = read_file.readline()
                    if line:
                        f.write(line)
                        f.flush()
                    else:
                        read_files.remove(read_file)
                        read_file.close()
            for temp_path in temp_paths:
                os.remove(temp_path)

    @staticmethod
    def _save_to_train_val(train_path: str, val_path, train_val_ratio: int, gen_list: list, total_num: int,
                           batch_size: int, thread_idx: int, temp_dir: str) -> None:
        if total_num <= batch_size:
            container = []
            for _, gen in gen_list:
                container.extend([case for case in gen])
            random.shuffle(container)
            split_idx = int(total_num / train_val_ratio)
            Util._save_cases(val_path, container[:split_idx])
            Util._save_cases(train_path, container[split_idx:])
        else:
            temp_paths, read_files = \
                Util._temp_save_to_disk(gen_list, temp_dir, total_num, batch_size, thread_idx)
            f_train = open(train_path, 'w')
            f_val = open(val_path, 'w')
            count = 0
            while len(read_files) > 0:
                read_file = random.choice(read_files)
                line = read_file.readline()
                if line:
                    if count % train_val_ratio == 1:
                        f_val.write(line)
                        f_val.flush()
                    else:
                        f_train.write(line)
                        f_train.flush()
                else:
                    read_files.remove(read_file)
                    read_file.close()
                count += 1
            f_train.close()
            f_val.close()
            for temp_path in temp_paths:
                os.remove(temp_path)

    @staticmethod
    def _temp_save_to_disk(gen_list, save_dir, total_num, batch_size, thread_idx=None) -> tuple:
        batch_num = math.ceil(total_num / batch_size)
        adjust_bs = math.ceil(total_num / batch_num)
        gen_list_split = []
        for _ in range(batch_num):
            gen_batch = []
            count = 0
            for i, (num, gen) in enumerate(gen_list):
                if count >= adjust_bs:
                    break
                if count + num <= adjust_bs:
                    gen_batch.append(gen)
                    gen_list.pop(i)
                    count += num
                else:
                    gen_1, gen_2 = tee(gen, 2)
                    gen_batch.append(islice(gen_1, 0, adjust_bs - count))
                    num = num - (adjust_bs - count)
                    if num > 0:
                        gen_list[i] = ((num), islice(gen_2, adjust_bs - count))
                    else:
                        gen_list.pop(i)
                    count += (adjust_bs - count)
            gen_list_split.append(gen_batch)
        temp_paths = []
        for i, gen_batch in enumerate(gen_list_split):
            container = []
            for gen in gen_batch:
                container.extend([case for case in gen])
            random.shuffle(container)
            if thread_idx is not None:
                temp_path = os.path.join(save_dir, f'temp{thread_idx}_{i}.txt')
            else:
                temp_path = os.path.join(save_dir, f'temp{i}.txt')
            Util._save_cases(temp_path, container)
            temp_paths.append(temp_path)
        read_files = [open(temp_path, 'r') for temp_path in temp_paths]
        return temp_paths, read_files
    
@dataclass
class StandardCaseConfig:
    min_num: int = field(default=1)
    max_num: int = field(default=1)
    mdf_cs_min_num: int = field(default=1)
    mdf_cs_max_num: int = field(default=1)
    sample_try_times:int = field(default=20)
    op_max_len: int = field(default=6)
    len1_prob: float = field(default=0.2)
    len2_prob: float = field(default=0.3)
    len3_prob: float = field(default=0.3)
    inst_min_len: int = field(default=6)
    multiple_tp_ratio:float = field(default=0.1)
    tp_max_per_trait:int = field(default=2)

    dropout: bool = field(default=True)
    dropout_ratio: float = field(default=0.05)
    add_noise: bool = field(default=True)
    max_noise_num: int = field(default=3)
    min_noise_num: int = field(default=0)
    noise_try_times: int = field(default=3)

    common_prob: float = field(default=0.1)
    max_common: int = field(default=2)
    add_psc: bool = field(default=True)
    add_med: bool = field(default=True)

    add_accessory: bool = field(default=True)

    align_instance: bool = field(default=True)
    status_tokens_last: bool = field(default=True)
    tong_pul_swift: float = field(default=0.2)
    tong_pul_first: float = field(default=0.05)
    move_accessory: bool = field(default=False)

    synonym_transforming:bool = field(default=True)

class StandardCase(StandardCaseConfig):
    def __init__(self, necessary_root_elem_li, root_elements, contradictory, noise_pool, modularized, synonym,
                 accessory, hierarchical_accessory):
        assert self.op_max_len >= 4 or self.len1_prob + self.len2_prob + self.len3_prob >= 1.0
        self.necessary_root_elem_li = necessary_root_elem_li
        self.root_elements = root_elements
        self.contradictory = contradictory
        self.noise_pool = noise_pool
        self.modularized, self.module_cls_paths, self.mdl_tong_body, self.mdl_tong_coating, self.mdl_pul = modularized
        self.special_token_paths = ['饮食', '睡眠', '大便', '小便']
        self.synonym = synonym
        self.accessory = accessory
        self.hierarchical_accessory = hierarchical_accessory
        self.len1_interval = [0, self.len1_prob]
        self.len2_interval = [self.len1_prob, self.len1_prob + self.len2_prob]
        self.len3_interval = [self.len1_prob + self.len2_prob, self.len1_prob + self.len2_prob + self.len3_prob]

        psc_num, rule_num, cls, cpl, opt, tong_pul, psc, med = self.root_elements
        self.psc_num = psc_num
        self.rule_num = rule_num
        self.cls = self._prepare_cls(cls)
        self.psc = self._prepare_psc(psc)
        self.med = self._prepare_med(med)
        self.cpl = self._prepare_compulsory(cpl)
        self.opts, self.n_optionals_with_null = self._prepare_optionals(opt)
        self.tong_pul, self.tp_cpl_lib, self.tp_opt_lib, self.tp_depend = self._prepare_tongue_pulse(tong_pul)

        self.dropout_elems_list = self._get_dropout_elems()
        self.dropout = self.dropout if len(self.dropout_elems_list) > 0 else False

        self.num_to_generate = self._get_num_to_generate()
        self.compulsory_gen = self._sample_compulsory()
        self.optional_gen = self._sample_optionals()
        self.tong_pul_gen = self._sample_tong_puls(self.multiple_tp_ratio, self.tp_max_per_trait)
        self.common_gen = self._sample_special(opt[-1], self.max_common, self.common_prob)

    def _get_num_to_generate(self):
        xs = []
        if self.cpl:
            xs.append(len(self.cpl))
        if self.opts:
            xs.append(len(set(chain.from_iterable(self.opts))))
        if self.tong_pul:
            xs.append(len(set(chain.from_iterable(self.tong_pul.values()))))
        k = max(xs)
        if re.search(r'[Bd+&*]*',self.rule_num):
            return min(max(2**k, self.mdf_cs_min_num), self.mdf_cs_max_num)
        return min(max(2**k, self.min_num), self.max_num)

    def _sample_and_check(self, gen, existing=None):
        for _ in range(self.sample_try_times):
            content = set() if existing is None else copy.deepcopy(existing)
            content.update(next(gen))
            if not self._is_contradictory(content):
                return content, False
        return existing, True

    def _sample_compulsory(self):
        while True:
            content = random.choice(self.cpl)
            if self.add_accessory:
                content = self._add_accessory(content)
            yield content

    def _sample_optionals(self):
        while True:
            content = set()
            for optional in self.opts:
                flt = random.random()
                if Util._in_range(flt, *self.len1_interval):
                    k = min(1, len(optional))
                elif Util._in_range(flt, *self.len2_interval):
                    k = min(2, len(optional))
                elif Util._in_range(flt, *self.len3_interval):
                    k = min(3, len(optional))
                else:
                    if self.op_max_len >= 4:
                        if 'null' in optional:
                            k = min(random.randint(4, max(self.op_max_len, self.n_optionals_with_null)), len(optional))
                        else:
                            k = min(random.randint(4, self.op_max_len), len(optional))
                    else:
                        raise Exception
                content.update(list(np.random.choice(optional, k, replace=False)))
            content = set(chain.from_iterable(x.split('&') for x in content))
            content = set(random.choice(x.split('/')) for x in content)
            if self.add_accessory:
                content = self._add_accessory(content)
            yield content

    def _sample_special(self, pool:Union[list, dict], max_num:int=1, prob:float=1.0):
        if all((bool(prob), bool(pool))):
            return self._sample_special_(pool, max_num, prob)
        else:
            return None

    @classmethod
    def _sample_special_(cls, pool:Union[list, dict], max_num:int=1, prob:float=1.0):
        assert max_num >= 1
        if isinstance(pool, list):
            while True:
                if random.random() < prob:
                    k = random.randint(1, min(max_num, len(pool)))
                    elems = list(np.random.choice(pool, k, replace=False))
                    yield elems
                else:
                    yield []
        elif isinstance(pool, dict):
            while True:
                elems = set()
                for lib in pool.keys():
                    if re.search(r'<[cC]>', lib):
                        elems.update(pool[lib])
                    elif re.search(r'<[oO][a-zA-Z]*>', lib):
                        k = random.randint(1, min(max_num, len(pool)))
                        elems.update(list(np.random.choice(pool[lib], k, replace=False)))
                    elif lib == 'own':
                        if random.random() < prob:
                            k = random.randint(1, min(max_num, len(pool)))
                            elems.update(list(np.random.choice(pool['<own>'], k, replace=False)))
                    else:
                        raise Exception
                yield elems
        else:
            raise Exception

    def _sample_tong_puls(self, multiple_tp_ratio:float = 0.3, max_per_trait:int=2):
        multiple_elem_dict = {'t_color': True, 't_nature': False, 't_coating_thickness': True,
                                't_coating_color': True, 't_coating_humidity': False, 't_coating_character': False,
                                'p_rate': False, 'p_rhythm': False, 'p_position': True, 'p_body': True,
                                'p_strength': True, 'p_fluency': True, 'p_tension': True, 'p_complex': False}
        assert max_per_trait >= 1
        multiple_elem_dict = {k : max_per_trait if v and random.random() < multiple_tp_ratio
        else 1 for k, v in multiple_elem_dict.items()}
        while True:
            remain_per_trait = multiple_elem_dict
            content = set()
            if self.tp_cpl_lib:
                content.update(chain.from_iterable(self.tp_cpl_lib.values()))
                for group_name in self.tp_cpl_lib.keys():
                    remain_per_trait[group_name] -= 1
            if self.tp_opt_lib:
                for tp_op in self.tp_opt_lib.values():
                    group_name = random.choice(list(tp_op.keys()))
                    content.update(np.random.choice(list(tp_op[group_name]), 1, replace=False))
                    remain_per_trait[group_name] -= 1
            for group_name in self.tong_pul.keys():
                if remain_per_trait[group_name] >= 1:
                    k = random.randint(1, min(remain_per_trait[group_name], len(self.tong_pul[group_name])))
                    content.update(np.random.choice(list(self.tong_pul[group_name]), k, replace=False))
            if self.tp_depend:
                independent_tps = content & set(self.tp_depend.keys())
                if independent_tps:
                    for independent_tp in independent_tps:
                        content.update([random.choice(xs.split('/')) for xs in self.tp_depend[independent_tp]])
            yield content

    @classmethod
    def _tong_pul_to_dict(cls, t_color, t_nature, t_coating_thickness, t_coating_color, t_coating_humidity,
                          t_coating_character, p_rate, p_rhythm, p_position, p_body, p_strength, p_fluency,
                          p_tension, p_complex):
        result = {'t_color': t_color, 't_nature': t_nature, 't_coating_thickness': t_coating_thickness,
                  't_coating_color': t_coating_color, 't_coating_humidity': t_coating_humidity,
                  't_coating_character': t_coating_character, 'p_rate': p_rate, 'p_rhythm': p_rhythm,
                  'p_position': p_position, 'p_body': p_body, 'p_strength': p_strength,
                  'p_fluency': p_fluency, 'p_tension': p_tension, 'p_complex': p_complex}
        return result

    def _prepare_cls(self, cls):
        return cls.split('&')

    def _prepare_compulsory(self, compulsory):
        content = list(filter(lambda x: len(x) >= 1, compulsory))
        if self.cls and self.cls != ['ALL']:
            if not content:
                content.append(self.cls)
            else:
                content = [cpl + self.cls for cpl in content if not set(cpl).issuperset(self.cls)]
        return content if len(content) >= 1 else None

    def _prepare_optionals(self, optionals):
        temp = list(filter(lambda x: x is not None and len(x) > 0, optionals[:-1]))
        optionals_with_null = [elems_group for elems_group in temp if 'null' in elems_group]
        n_optionals_with_null = len(optionals_with_null)
        optionals_with_null = list(set(chain.from_iterable(optionals_with_null)))
        optional_list = list(filter(lambda x: 'null' not in x, temp))
        if optionals_with_null:
            optional_list.append(optionals_with_null)
        return (optional_list, n_optionals_with_null) if len(optional_list) >=1 else (None, None)

    def _prepare_tongue_pulse(self, tongue_pulses):
        tong_pul = self._tong_pul_to_dict(*tongue_pulses)
        tps = dict()
        tp_cpl_lib = dict()
        tp_op_lib = dict()
        tp_depend = dict()
        for group_name in tong_pul.keys():
            tps[group_name] = set()
            for data in tong_pul[group_name]:
                data_split = data.replace('; ', '；').replace(';', '；').split('；')
                if 'F' in data_split[0]:
                    tps[group_name].add('null')
                else:
                    tps[group_name].add(data_split[0])

                if data_split[1]:
                    for cop in data_split[1].split('&'):
                        if re.search(r'[Cc]', cop):
                            if group_name not in tp_cpl_lib.keys():
                                tp_cpl_lib[group_name] = set()
                            tp_cpl_lib[group_name].add(data_split[0])
                        elif re.search(r'[Oo]',cop):
                            if cop not in tp_op_lib.keys():
                                tp_op_lib[cop] = dict()
                            if group_name not in tp_op_lib[cop].keys():
                                tp_op_lib[cop][group_name] = set()
                            tp_op_lib[cop][group_name].add(data_split[0])
                        else:
                            raise Exception
                depend_elems = data_split[2].split('&')
                if '' in depend_elems:
                    depend_elems.remove('')
                if depend_elems:
                    tp_depend[data_split[0]] = depend_elems
        if len(set(chain.from_iterable(tps.values()))) >= 1:
            return tps, tp_cpl_lib, tp_op_lib, tp_depend
        else:
            return (None, ) * 4

    def _prepare_med(self, med):
        return [x.replace('；', '') for x in med]

    def _prepare_psc(self, psc):
        return [x.replace('；', '') for x in psc]
    def _add_accessory(self, sequence:Union[list, set]):
        if isinstance(sequence, list):
            sequence.extend(chain.from_iterable([self.accessory[x] for x in sequence if x in self.accessory.keys()]))
        elif isinstance(sequence, set):
            sequence.update(chain.from_iterable([self.accessory[x] for x in sequence if x in self.accessory.keys()]))
        else:
            raise Exception
        return sequence

    def _get_dropout_elems(self):
        content = {'sym_phy':list(), 'tong_pul':list()}
        _, _, cls, compulsories, optionals, tong_puls, psc, med = copy.deepcopy(self.root_elements)
        cpl_ctn = set()
        for compulsory in compulsories:
            if len(compulsory) == 0:
                continue
            if cls in compulsory:
                compulsory.remove(cls)
            cpl_ctn.update(compulsory)
        if cpl_ctn:
            content['sym_phy'].append(list(cpl_ctn))
        for optional in optionals[:-9]:
            if len(optional) == 0:
                continue
            if 'null' in optional:
                continue
            if cls in optional:
                optional.remove(cls)
            content['sym_phy'].append(optional)

        for tong_pul in tong_puls:
            tong_pul = [elem.split('；')[0] for elem in tong_pul]
            if 'null' in tong_pul:
                continue
            content['tong_pul'].append(tong_pul)
        return content

    def _dropout(self, case: dict[str:Union[list, set]]):
        if self.dropout_ratio < random.random():
            return case
        else:
            dropout_place = random.choice(list(self.dropout_elems_list.keys()))
            dropout_elems = random.choice(list(self.dropout_elems_list[dropout_place]))
            overlap_elems = set(case[dropout_place]) & set(dropout_elems)
            if len(overlap_elems) == 0:
                return case
            if len(case[dropout_place]) - len(overlap_elems) < self.inst_min_len:
                return case
            case[dropout_place] = list(set(case[dropout_place]) - overlap_elems)
            return case

    def _contradictory_elems_filter(self, obj_1, obj_2):
        content = set()
        for x in obj_2:
            if x in self.contradictory:
                if bool(set(obj_1) & set(self.contradictory[x])):
                    content.add(x)
        return list(set(obj_2) - content)

    def _is_contradictory(self, inst):
        for x in inst:
            if x in self.contradictory:
                if bool(set(inst) & set(self.contradictory[x])):
                    return True
        return False

    def _add_tokens(self, case: dict[str:Union[list, set]], tokens:Union[list, set], place:str):
        if place not in case.keys():
            case[place] = set()
        if self.add_accessory:
            tokens = self._add_accessory(tokens)
        tokens = list(filter(lambda x: x not in case, set(tokens)))
        tokens = self._contradictory_elems_filter(tokens, tokens)
        tokens = self._contradictory_elems_filter(case[place], tokens)
        if isinstance(case[place], list):
            case[place].extend(tokens)
        elif isinstance(case[place], set):
            case[place].update(tokens)
        else:
            raise Exception
        return case

    def _sample_noises(self, case):
        assert len(case) > 0 and len(self.noise_pool.keys()) >= self.min_noise_num
        chained_case = list(chain.from_iterable(case.values()))
        max_num = min(math.floor(len(chained_case) / 2), math.floor(len(case['sym_phy']) / 2),
                      self.max_noise_num, len(self.noise_pool.keys()))
        min_num = self.min_noise_num
        assert max_num >= min_num
        k = min(random.randint(min_num, max_num), len(self.noise_pool.keys()))
        noises = []
        for _ in range(self.noise_try_times):
            noises = list(np.random.choice(list(self.noise_pool.keys()), k, replace=False))
            new_inst = set(chained_case + noises)
            valid_flag = True
            if not noises:
                break
            for necessary_root_elems in self.necessary_root_elem_li:
                next_flag = True
                if necessary_root_elems[1] == self.rule_num:
                    continue
                for compulsory_check_items in necessary_root_elems[3]:
                    if not new_inst.issuperset(set(compulsory_check_items)):
                        next_flag = True
                        continue
                    else:
                        next_flag = False
                        break
                if next_flag:
                    continue

                for optional_check_items in necessary_root_elems[4] + necessary_root_elems[5]:
                    if not new_inst & set(optional_check_items):
                        next_flag = True
                        break
                    else:
                        elems_necessary_combi = set(necessary_root_elems[6].keys()) & set(optional_check_items)
                        if elems_necessary_combi:
                            for elem in elems_necessary_combi:
                                if not new_inst.issuperset(necessary_root_elems[6][elem]):
                                    next_flag = True
                                else:
                                    next_flag = False
                                    break
                        else:
                            next_flag = False
                if next_flag:
                    continue
                due_to_inst = True
                if due_to_inst:
                    for compulsory_check_items in necessary_root_elems[3]:
                        if set(case).issuperset(set(compulsory_check_items)):
                            due_to_inst = True
                            break
                        else:
                            due_to_inst = False
                            continue
                if due_to_inst:
                    for optional_check_items in necessary_root_elems[4] + necessary_root_elems[5]:
                        if not set(case) & set(optional_check_items):
                            due_to_inst = False
                            break
                        else:
                            elems_necessary_combi = set(necessary_root_elems[6].keys()) & set(optional_check_items)
                            if elems_necessary_combi:
                                to_break = False
                                for elem in elems_necessary_combi:
                                    if not set(case).issuperset(necessary_root_elems[6][elem]):
                                        due_to_inst = False
                                        to_break = True
                                        break
                                if to_break:
                                    break
                if due_to_inst:
                    continue
                valid_flag = False
                break
            if valid_flag:
                break
            noises = []
        content = dict()
        for noise in noises:
            if self.noise_pool[noise] not in content:
                content[self.noise_pool[noise]] = []
            content[self.noise_pool[noise]].append(noise)
        return content

    def _align_instance(self, inst:Union[list, set]):
        inst = {elem: self.modularized[elem] for elem in inst if elem in self.modularized}
        head_module_paths = {self.modularized[elem][1] for elem in self.cls if elem in self.modularized}
        head_related_tokens = dict()
        status_tokens = dict()
        other_tokens = dict()
        for elem in inst.keys():
            if elem in self.cls and elem not in inst.keys():
                if 'ukn' not in head_related_tokens.keys():
                    head_related_tokens['ukn'] = set()
                head_related_tokens['ukn'].add(elem)
            if 'ALL' not in self.cls and inst[elem][1] in head_module_paths:
                if inst[elem][1] not in head_related_tokens.keys():
                    head_related_tokens[inst[elem][1]] = set()
                head_related_tokens[inst[elem][1]].add(elem)
                continue
            if self.status_tokens_last and inst[elem][1] in self.special_token_paths:
                if inst[elem][1] not in status_tokens.keys():
                    status_tokens[inst[elem][1]] = set()
                status_tokens[inst[elem][1]].add(elem)
                continue
            if inst[elem][1] not in other_tokens.keys():
                other_tokens[inst[elem][1]] = set()
            other_tokens[inst[elem][1]].add(elem)

        if self.add_accessory and self.move_accessory:
            pass

        content = []
        if head_related_tokens:
            content.extend(chain.from_iterable(head_related_tokens.values()))
        if other_tokens:
            other_tokens = {k: other_tokens[k] for k in
                            sorted(other_tokens, key=lambda x: len(other_tokens[x]), reverse=True)}
            content.extend(chain.from_iterable(other_tokens.values()))
        if status_tokens:
            content.extend(chain.from_iterable(status_tokens.values()))
        return content

    def _align_tong_pul(self, inst: Union[list, set]):
        inst = set(inst)
        content = {'舌质':[], '舌苔':[], '脉象':[]}
        for traits in self.mdl_tong_body.values():
            content['舌质'].extend(inst & traits)
        for traits in self.mdl_tong_coating.values():
            content['舌苔'].extend(inst & traits)
        for traits in self.mdl_pul.values():
            content['脉象'].extend(inst & traits)
        content = [content['舌质'] + content['舌苔'], content['脉象']]
        if random.random() < self.tong_pul_swift:
            content.reverse()
        content = list(chain.from_iterable(content))
        return content

    def _move_accessory(self, content, elem, hierarchical_accessory: dict, i: int = 0):
        idx_dict = dict()
        if elem in hierarchical_accessory.keys():
            elems_to_move = list(set(content) & set(hierarchical_accessory[elem].keys()))
            shuffle(elems_to_move)
            content = [x for x in content if x not in elems_to_move]
            for elem_to_move in elems_to_move:
                i += 1
                idx_dict[elem_to_move] = i
                content.insert(content.index(elem) + i, elem_to_move)
            for elem_to_move in elems_to_move:
                if len(hierarchical_accessory[elem][elem_to_move].keys()) > 0:
                    content = self._move_accessory(content, elem_to_move, hierarchical_accessory[elem],
                                             i - idx_dict[elem_to_move])
        return content

    def _synonym_transforming(self, inst: Union[list, set]):
        return [random.choice(self.synonym[elem]) if elem in self.synonym.keys() else elem for elem in inst]

    def _get_case_generator(self):
        while self.num_to_generate:
            break_flag = False
            case = dict()
            sym_phy = set()
            if self.cpl is None and self.opts is None and self.tong_pul is None:
                logger.warn(f'方剂编号{self.psc_num}, 规则编号{self.rule_num}内容为空，予跳过')
                break
            if self.cpl:
                sym_phy.update(next(self.compulsory_gen))
            if self.opts:
                sym_phy, break_flag = self._sample_and_check(self.optional_gen, sym_phy)
            if sym_phy:
                case['sym_phy'] = sym_phy
            if self.tong_pul:
                tong_pul, break_flag = self._sample_and_check(self.tong_pul_gen)
                case['tong_pul'] = tong_pul
            if break_flag:
                logger.warn(f'方剂编号{self.psc_num}, 规则编号{self.rule_num}规则矛盾，该规则生成跳过')
                break
            case = {k: list(filter(lambda x: x != 'null', v)) for k, v in case.items()}
            if self.dropout:
                case = self._dropout(case)
            if self.common_gen:
                common_indications = next(self.common_gen)
                case = self._add_tokens(case, common_indications, 'sym_phy')
            if not any([case[place] for place in ['sym_phy', 'tong_pul', 'disease'] if place in case]): # 若全为空，则跳过
                continue
            if self.add_noise:
                noises = self._sample_noises(case)
                for place, noise in noises.items():
                    case = self._add_tokens(case, noise, place)
            case = {k: list(set(filter(lambda x: x != 'null', v))) for k, v in case.items()}
            if self.align_instance:
                case['sym_phy'] = self._align_instance(case['sym_phy'])
                case['tong_pul'] = self._align_tong_pul(case['tong_pul'])
            if self.synonym_transforming:
                case['sym_phy'] = self._synonym_transforming(case['sym_phy'])
                case['tong_pul'] = self._synonym_transforming(case['tong_pul'])
            if self.cls:
                case['cls'] = self.cls
            if self.add_psc and self.psc:
                case['psc'] = self.psc
            if self.add_med and self.med:
                case['med'] = self.med
            yield case
            self.num_to_generate -= 1

    def get_case_generator_wrapper(self):
        return self._case_generator_wrapper(), self.num_to_generate

    def _case_generator_wrapper(self):
        case_gen = self._get_case_generator()
        for case in case_gen:
            line = str(case['cls']) + '\t' + str(case['sym_phy'] + case['tong_pul']) + '\t' + str(case['psc']) + '\t' + str(case['med']) + '\n'
            line = line.replace('[', '').replace(']', '').replace("'", '')
            yield line

@dataclass
class CaseGenerator:
    generator_name: str
    root_elems_list: list
    contradictory: dict
    noise_pool: dict
    synonym: dict
    modularized: tuple
    accessory: dict
    future_path_num: int = field(default=50)
    batch_size:int = field(default=30000000)

    def __post_init__(self):
        self.necessary_root_elem_li = Util._get_necessary_root_elem_list(self.root_elems_list)
        self.hierarchical_accessory = Util._get_hierarchical_accessory(self.accessory)
        self._standard_dir = os.path.join(self.generator_name, 'standard')
        self._temp_dir = os.path.join(self.generator_name, 'temp')
        self.std_train_dir = os.path.join(self._standard_dir, 'trainData')
        self.std_val_dir = os.path.join(self._standard_dir, 'valData')
        self.std_test_dir = os.path.join(self._standard_dir, 'testData')
        self.future_path_num = self.future_path_num
        self.std_train_paths = [os.path.join(self.std_train_dir, f'trainData{i}.txt')
                                for i in range(self.future_path_num)]
        self.std_val_paths = [os.path.join(self.std_val_dir, f'valData{i}.txt')
                              for i in range(self.future_path_num)]
        Util._make_dir(self.std_train_dir, self.std_val_dir, self.std_test_dir,
                       self._temp_dir)

    def generate_cases(self, save_path: str, step: int = 1) -> None:
        self._generate_standard_cases(save_path, step)

    def generate_train_val_cases(self, N_times: int = 1, step: int = 1, begin_num: int = 0,
                                 train_val_ratio: int = 10) -> None:
        self._generate_standard_train_val_cases(N_times, step, begin_num, train_val_ratio)

    def _generate_standard_cases(self, save_path: str, step: int = 1) -> None:

        gen_list = []
        total_num = 0
        for root_elements in self.root_elems_list:
            num, gen = self._make_standard_case_generator(root_elements, step)
            if num == 0:
                continue
            total_num += num
            gen_list.append((num, gen))
        Util._save_to_one_dataset(save_path, gen_list, total_num, self.batch_size)

    def _generate_standard_train_val_cases(self, N_times: int = 1, step: int = 1, begin_num: int = 0,
                                           train_val_ratio: int = 10) -> tuple:
        train_paths, val_paths = [], []
        for i in range(0, N_times):
            gen_list = []
            total_num = 0
            for root_elements in self.root_elems_list:
                num, gen = self._make_standard_case_generator(root_elements, step)
                if num == 0:
                    continue
                total_num += num
                gen_list.append((num, gen))
            train_path = os.path.join(self.std_train_dir, f'trainData{i + begin_num}.txt')
            val_path = os.path.join(self.std_val_dir, f'valData{i + begin_num}.txt')
            train_paths.append(train_path)
            val_paths.append(val_path)
            Util._save_to_train_val(train_path, val_path, train_val_ratio, gen_list, total_num,
                                    self.batch_size, i, self._temp_dir)
        return train_paths, val_paths

    def _make_standard_case_generator(self, root_elements, step):
        inst = StandardCase(self.necessary_root_elem_li, root_elements, self.contradictory, self.noise_pool,
                            self.modularized, self.synonym, self.accessory, self.hierarchical_accessory)
        case_generator, n_case = inst.get_case_generator_wrapper()
        if step > 1:
            n_case = math.floor(n_case / step)
            case_generator = islice(case_generator, 0, None, step)
        return n_case, case_generator


if __name__ == '__main__':
    program_name = 'Gen'
    adm = SQLAdministrator(program_name)
    adm.init_sql()
    adm.save_root_data(os.path.join('root_data', 'root_mod.xlsx'),
                       os.path.join('root_data', 'vocab.xlsx'))
    cg = CaseGenerator(generator_name= program_name,
                       root_elems_list= adm.root.get_rule_element_list(),
                       contradictory= adm.root.get_contradictory(),
                       noise_pool = adm.root.get_noises(),
                       synonym=adm.root.get_synonym(),
                       modularized=adm.root.get_modularized(),
                       accessory=adm.root.get_accessory())
    cg.generate_train_val_cases(N_times=1, begin_num=0, train_val_ratio=10)
