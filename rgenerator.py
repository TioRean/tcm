import math
import numpy as np
import copy
import pandas as pd
import os
import warnings
import re
from itertools import combinations, chain, product
from logger import get_logger

logger = get_logger('rgenerate')

class RuleGenerator:
    def __init__(self, org_file_path, head_list:list = None, drop_info_list: list = None):
        '''
        根据制定的规则，生成新的中医方证规则。
        :param org_file_path: 中医方证规则的原始模板路径，一般该文件存储了所有的中医方证规则
        :param head_list: 用于指定目标规则的Head，若无则默认head_list进行生成
        :param drop_info_list: 用于指定删除元素，若无则默认drop_list进行生成
        '''
        self.tp_group_traits = {'t_color':('舌淡白', '舌淡暗', '舌淡红', '舌暗红', '舌红', '舌绛', '舌紫', '舌青'),
                               't_nature':('舌老','舌嫩','null'),
                               't_coating_color':('白苔', '黄苔', '灰苔', '黑苔'),
                               't_coating_thickness':('苔少','苔薄','苔厚'),
                               't_coating_humidity':('苔水滑', '苔燥','null.1'),
                               't_coating_character':('苔腻','苔腐','null.2'),
                               'p_rate':('脉数', '脉疾', '脉迟','null.3'),
                               'p_rhythm':('脉促','脉结','脉代','null.4'),
                               'p_position':('脉浮', '脉沉', '脉伏','null.5'),
                               'p_body':('脉大', '脉细', '脉长', '脉短','null.6'),
                               'p_strength':('脉虚', '脉弱', '脉微', '脉实', '脉弹指', '无脉','null.7'),
                                'p_fluency':('脉滑', '脉涩','null.8'),
                                'p_tension':('脉弦', '脉紧', '脉缓', '脉硬', '脉软','null.9'),
                                'p_complex':('革脉', '牢脉', '洪脉', '动脉', '芤脉', '浮大中空脉', '濡脉','null.10')}
        self.tp_traits_group = self._get_reverse_dict(self.tp_group_traits)
        self.org_file_path = org_file_path
        self.df = pd.read_excel(self.org_file_path, sheet_name='规则').fillna('null')
        try:
            self.mdf_df = pd.read_excel(self.org_file_path, sheet_name='加减法').fillna('NA')
        except ValueError:
            self.mdf_df = None
        self._prepare_data()
        self.head_list = head_list if head_list is not None else \
            ['发热','意识障碍', '头痛', '胸痛', '腹泻', '呕吐', '中腹痛', '经期腹痛', '形神倦怠', '上腹痛']
        self.drop_info_list = drop_info_list if drop_info_list is not None else \
                ['drop_tong', 'drop_pul', 'drop_tong_pul']
    def _prepare_data(self):
        for i, row in self.df.iterrows():
            for column in row.keys():
                self.df.loc[i, column] = re.sub(r'[ ]*[;；][ ]*', '；', self.df.loc[i, column])
                self.df.loc[i, column] = self.df.loc[i, column].strip(',.;，。 ')
        for i, row in self.df.loc[:,'舌淡白':'null.10'].iterrows():
            for column in row.keys():
                if 'T' in self.df.loc[i, column]:
                    temp = self.df.loc[i, column].split('；')
                    while len(temp) < 3:
                        temp.append('')
                    self.df.loc[i, column] = '；'.join(temp)
        if self.mdf_df is not None:
            for i, row in self.mdf_df.iterrows():
                for column in row.keys():
                    self.mdf_df.loc[i, column] = re.sub(r'[ ]*[;；][ ]*', '；', self.mdf_df.loc[i, column])
                    if re.search(r'[Oo][A-Za-z]+', self.mdf_df.loc[i, column]):
                        temp = self.mdf_df.loc[i, column].split('；')
                        temp[1] = temp[1].lower()
                        self.mdf_df.loc[i, column] = '；'.join(temp)
                    self.mdf_df.loc[i, column] = self.mdf_df.loc[i, column].strip(',.;，。 ')

            for i, row in self.mdf_df.loc[:,'舌淡白':'null.10'].iterrows():
                for column in row.keys():
                    if 'T' in self.mdf_df.loc[i, column]:
                        temp = self.mdf_df.loc[i, column].split('；')
                        while len(temp) < 3:
                            temp.append('')
                        self.mdf_df.loc[i, column] = '；'.join(temp)

    @staticmethod
    def _get_reverse_dict(dict_obj:dict[str:list]):
        reverse_dict = dict()
        for k, vs in dict_obj.items():
            for v in vs:
                reverse_dict[v] = k
        return reverse_dict

    @classmethod
    def _get_new_file_path(cls, org_file_path, new_path_name:str):
        org_file_name, suffix = os.path.splitext(os.path.basename(org_file_path))
        new_file_path = os.path.join(os.path.dirname(org_file_path), org_file_name + f'_{new_path_name}' + suffix)
        return new_file_path

    def make_drop_info_rules(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        org_df = pd.read_excel(self.org_file_path)
        new_file_paths = []
        for item in self.drop_info_list:
            if item == 'drop_tong':
                new_file_path = RuleGenerator._get_new_file_path(self.org_file_path, item)
                temp_df = copy.copy(org_df)
                temp_df.loc[:, '舌淡白':'null.2'] = None
                temp_df.loc[:, '类别'] += '_drop_tong'
                temp_df.to_excel(new_file_path, sheet_name='Sheet1', index=False)
                new_file_paths.append(new_file_path)
            elif item == 'drop_pul':
                new_file_path = RuleGenerator._get_new_file_path(self.org_file_path, item)
                temp_df = copy.copy(org_df)
                temp_df.loc[:, '脉数':'null.10'] = None
                temp_df.loc[:, '类别'] += '_drop_pul'
                temp_df.to_excel(new_file_path, sheet_name='Sheet1', index=False)
                new_file_paths.append(new_file_path)
            elif item == 'drop_tong_pul':
                new_file_path = RuleGenerator._get_new_file_path(self.org_file_path, item)
                temp_df = copy.copy(org_df)
                temp_df.loc[:, '舌淡白':'null.10'] = None
                temp_df.loc[:, '类别'] += '_drop_tong_pul'
                temp_df.to_excel(new_file_path, sheet_name='Sheet1', index=False)
                new_file_paths.append(new_file_path)
        return new_file_paths

    def _add_num(self, save_path):
        rule_df = pd.read_excel(self.org_file_path, sheet_name='规则')
        df = copy.deepcopy(rule_df)
        mdf_df = pd.read_excel(self.org_file_path, sheet_name='加减法')
        num_df = pd.read_excel(self.org_file_path, sheet_name='方剂编号')
        num_df = num_df.loc[:, '方名':'方剂编号']
        num_dict = num_df.to_dict('list')
        num_pairs = list(zip(num_dict['方名'], num_dict['方剂编号']))
        number_dict = dict()
        for pair in num_pairs:
            number_dict[pair[0]] = pair[1]

        for i, row in rule_df.iterrows():
            rule_df.loc[i, '方剂编号'] = number_dict[row['方名']]
        rule_df.to_excel(save_path, sheet_name='规则', index=True)

    def _generate_head_rules(self, split:bool=False):
        def _get_head_df(head, org_df):
            df = copy.deepcopy(org_df)
            temp_df = df.loc[:, '类别':'参考资料']
            drop_i = []
            for i, row in temp_df.iterrows():
                keep_rule = False
                head_elem_column = None
                for idx, item in enumerate(row):
                    try:
                        if math.isnan(item):
                            continue
                    except TypeError:
                        pass
                    if head in item.split('；'):
                        keep_rule = True
                        head_elem_column = temp_df.columns[idx]
                if not keep_rule:
                    drop_i.append(i)
                else:
                    df.loc[i, '类别'] = head
                    if head_elem_column in row['必要元素库A':'必要元素库J'].to_dict().keys():
                        for pool in row['必要元素库A':'必要元素库J'].to_dict().keys():
                            try:
                                if math.isnan(df.loc[i, pool]):
                                    continue
                            except TypeError:
                                pass
                            if head not in df.loc[i, pool]:
                                df.loc[i, pool] = None
                    else:
                        for pool in row['必要元素库A':'必要元素库J'].to_dict().keys():
                            try:
                                if math.isnan(df.loc[i, pool]) and pool == '必要元素库A':
                                    df.loc[i, pool] = head
                            except TypeError:
                                if head not in df.loc[i, pool].split('；'):
                                    df.loc[i, pool] += '；' + head
                        new_opt = '；'.join(filter(lambda x: x != head, row[head_elem_column].split('；')))
                        if 'null' not in new_opt:
                            new_opt += '；null'
                        df.loc[i, head_elem_column] = new_opt
            df = df.drop(drop_i)
            return df

        self.df = pd.read_excel(self.org_file_path)
        if split:
            new_dfs = dict()
            for head in self.head_list:
                new_df = _get_head_df(head, self.df)
                new_df = new_df.fillna('null')
                new_dfs[head] = new_df
            return new_dfs
        else:
            df = copy.deepcopy(self.df)
            for head in self.head_list:
                new_df = _get_head_df(head, self.df)
                df = pd.concat([df, new_df], axis=0, ignore_index=True)
            df = df.fillna('null')
            return df

    def _generate_modification(self):
        def _get_modifications():
            content = dict()
            for i, row in self.mdf_df.iterrows():
                if row['方剂编号'] not in content.keys():
                    content[row['方剂编号']] = dict()
                if row['方名'] not in content[row['方剂编号']]:
                    content[row['方剂编号']]['方名'] = row['方名']
                else:
                    assert content[row['方剂编号']]['方名'] == row['方名']
                content[row['方剂编号']][row['加减法编号']] = row['合方': 'null.10']
            return content

        def _build_bracket_tree(text):
            root = {"content": text, "type": "", "children": []}
            stack = []
            brackets = {')': '(', ']': '[', '}': '{', '>': '<'}
            open_brackets = set(brackets.values())

            for i, char in enumerate(text):
                if char in open_brackets:
                    node = {"content": "", "type": "", "children": []}
                    stack.append((node, i + 1, char))
                elif char in brackets:
                    if not stack:
                        continue
                    current_node, start_pos, start_char = stack[-1]
                    if brackets[char] != start_char:
                        continue
                    stack.pop()
                    current_node["content"] = text[start_pos:i]
                    current_node["type"] = f"{start_char}{char}"
                    if stack:
                        parent_node, _, _ = stack[-1]
                        parent_node["children"].append(current_node)
                    else:
                        root["children"].append(current_node)
            return root

        def _combi(node):
            content = []
            xs = node['content'].replace(', ', ',').split(',')
            if node['type'] == '()':
                for i in range(1, 2):
                    for x in combinations(xs, i):
                        content.append('&'.join(x))
            elif node['type'] == '[]':
                for i in range(1, 3):
                    for x in combinations(xs, i):
                        content.append('&'.join(x))
            elif node['type'] == '{}':
                for i in range(1, 4):
                    for x in combinations(xs, i):
                        content.append('&'.join(x))
            elif node['type'] == '<>':
                for i in range(1, 5):
                    for x in combinations(xs, i):
                        content.append('&'.join(x))
            else:
                raise Exception
            return content

        def _process_modifications(text):
            text = re.sub(r'''[ ]*['"‘“]+[ ]*['"‘“]+[ ]*''', '', text)
            text = re.sub(r'[ ]*&[ ]*', '&', text)
            text = re.sub(r'[ ]*[（(][ ]*', '(', text)
            text = re.sub(r'[ ]*[）)][ ]*', ')', text)
            text = re.sub(r'[,，]+[ ]*', ',', text)
            tree = _build_bracket_tree(text)
            position = {}
            for i, node in enumerate(tree['children']):
                position[f'pos_{i}'] = _combi(node)
                tree['content'] = tree['content'].replace(node['type'][0] + node['content'] + node['type'][1],
                                                          f'pos_{i}')
            tree['content'] = re.sub(r'[,，]+[ ]*]', ',', tree['content']).split(',')
            content = set()
            for section in tree['content']:
                xs = section.split('&')
                for i, x in enumerate(xs):
                    if re.match(r'pos_', x):
                        xs[i] = position[x]
                    elif re.match(r'B\d+', x):
                        xs[i] = [x]
                    elif x == '':
                        xs[i] = ['']
                    else:
                        raise Exception
                for x in product(*xs):
                    content.add('&'.join(set(x)))
            content = [set(xs.split('&')) for xs in content if xs != '']
            executions = []
            for xs in content:
                save_flag = True
                for ys in executions:
                    if not xs - ys and not ys - xs:
                        save_flag = False
                        break
                if save_flag:
                    executions.append(xs)
            executions = [list(filter(lambda x: x != '', xs)) for xs in executions]
            return executions

        mdf_dict = _get_modifications()
        org_df, new_df = copy.deepcopy(self.df), copy.deepcopy(self.df)
        idx = 0
        for i, row in org_df.iterrows():
            new_df.loc[idx] = row
            psc_num = row['方剂编号']
            psc_name = row['方名（纲）']
            row_modif = copy.deepcopy(row)
            if row['A类加减法'] != 'null':
                row_modif['方剂编号'] = row['方剂编号'] + '-' +  row['A类加减法']
                row_modif['规则编号'] = row['规则编号'] + '-' + row['A类加减法']
            new_df.loc[idx] = copy.deepcopy(row_modif)
            idx += 1
            if row['B类加减法'] != 'null':
                executions = _process_modifications(str(row['B类加减法']))
                assert mdf_dict[psc_num]['方名'] == psc_name
                for execution in executions:
                    continue_flag = False
                    rule_num_mdf  = execution
                    psc_mdf = []
                    medic_mdf = {'加法':list(), '减法':list()}
                    cps_mdf = []
                    opt_mdf = {'备选元素库包含null': set()}
                    del_mdf = []
                    tp_mdf = {}
                    for i, mdf_num in enumerate(execution):
                        mdf = mdf_dict[psc_num][mdf_num]
                        adding_psc = str(mdf['合方']).split('&')
                        psc_mdf.extend(list(filter(lambda x: x not in psc_mdf and x != 'NA', adding_psc)))

                        adding_meds = str(mdf['加法']).split('；')
                        medic_mdf['加法'].extend(list(filter(lambda x: x not in medic_mdf['加法'] and x != 'NA',
                                                             adding_meds)))

                        dropping_meds = str(mdf['减法']).split('；')
                        medic_mdf['减法'].extend(list(filter(lambda x: x not in medic_mdf['减法'] and x != 'NA',
                                                             dropping_meds)))

                        cps_mdf.append([elements.split('；') for elements in mdf['必要元素库A':'必要元素库C']
                                        if 'NA' not in elements])

                        for optional in mdf['备选元素库A':'备选元素库O']:
                            optional = optional.split('；')
                            if 'null' in optional:
                                opt_mdf['备选元素库包含null'].update(optional)
                            elif 'NA' in optional:
                                continue
                            else:
                                opt_mdf[f'备选元素库{len(opt_mdf.keys())}'] = optional


                        del_mdf.extend([elem for elem in str(mdf['删除元素']).split('；') if elem != 'NA'])


                        new_tp_dict = dict()
                        for tp_group_name, tp_traits in self.tp_group_traits.items():
                            new_tp_dict[tp_group_name] = mdf[tp_traits[0]:tp_traits[-1]].to_dict()
                        for trait_group in new_tp_dict.keys():
                            if trait_group not in tp_mdf.keys():
                                for k, v in new_tp_dict[trait_group].items():
                                    if 'T' in v:
                                        new_tp_dict[trait_group][k] = '；'.join([x + f'_{i}' if j == 1 and x != '' else x
                                                                                for j, x in enumerate(v.split('；'))])
                                tp_mdf[trait_group] = new_tp_dict[trait_group]
                            else:
                                for trait in new_tp_dict[trait_group].keys():
                                    if new_tp_dict[trait_group][trait] == 'F':
                                        cnt = 0
                                        for change in tp_mdf[trait_group].values():
                                            if 'T' in change:

                                                cnt += 1
                                        if cnt == 1:

                                            if 'T' in tp_mdf[trait_group][trait]:
                                                continue_flag = True
                                                logger.warning(f"方名'{psc_name}' 方剂编号'{psc_num}', "
                                                              f"执行加减法编码{execution}舌脉加减法执行时存在矛盾")
                                                break
                                            else:
                                                tp_mdf[trait_group][trait] = 'F'
                                        else:
                                            tp_mdf[trait_group][trait] = 'F'
                                    elif new_tp_dict[trait_group][trait] == 'NA':
                                        continue
                                    else:
                                        if tp_mdf[trait_group][trait] == 'F':
                                            continue_flag = True
                                            logger.warning(f"方名'{psc_name}' 方剂编号'{psc_num}', "
                                                        f"执行加减法编码{execution}舌脉加减法执行时存在矛盾")
                                            break
                                        if tp_mdf[trait_group][trait] == 'NA':
                                            tp_mdf[trait_group][trait] = '；；'
                                        tp1 = [set(filter(lambda x: x != '', x.split('&')))
                                               for x in tp_mdf[trait_group][trait].split('；')]

                                        tp2 = [set(filter(lambda x: x!= '', x.split('&')))
                                               for x in new_tp_dict[trait_group][trait].split('；')]
                                        tp1[0].update(tp2[0])
                                        tp1[1].update([x + f'_{i}' for x in tp2[1]])
                                        tp2_nes = []
                                        for x in tp2[2]:
                                            temp = []
                                            for y in x.split('/'):
                                                if 'F' in tp_mdf[self.tp_traits_group[y]][y]:
                                                    continue
                                                temp.append(y)
                                            if temp:
                                                tp2_nes.append('/'.join(temp))
                                            else:
                                                continue_flag = True
                                                logger.warning(f"方名'{psc_name}' 方剂编号'{psc_num}', "
                                                               f"执行加减法编码{execution}舌脉加减法执行时存在矛盾")
                                        tp1[2].update(tp2_nes)
                                        tp_mdf[trait_group][trait] = '；'.join(['&'.join(xs) for xs in tp1])
                    if continue_flag:
                        continue

                    pdt_cps_mdf = []
                    for elements in product(*cps_mdf):
                        pdt_cps_mdf.append(list(chain.from_iterable(elements)))


                    if len(set(medic_mdf['加法']) & set(medic_mdf['减法'])) >= 1:
                        continue_flag = True
                        logger.warning(f"方名'{psc_name}' 方剂编号'{psc_num}', "
                                       f"执行加减法编码{execution}时药物加法与减法重叠")


                    for elements in pdt_cps_mdf:
                        if len(set(del_mdf) & set(elements)) > 0:
                            continue_flag = True
                            logger.warning(f"方名'{psc_name}' 方剂编号'{psc_num}', "
                                          f"执行加减法编码{execution}时必要元素加减法存在矛盾")

                    for elements in opt_mdf.values():
                        if len(elements) == 1:
                            if len(set(del_mdf) & set(elements)) > 0:
                                continue_flag = True
                                logger.warning(f"方名'{psc_name}' 方剂编号'{psc_num}', "
                                              f"执行加减法编码{execution}时备选元素加减法存在矛盾")

                    if set(del_mdf) & set(filter(lambda x: x != 'null', row['必要元素库A':'必要元素库J'].to_dict().values())):
                        continue_flag = True
                        logger.warning(f"方名'{psc_name}' 方剂编号'{psc_num}', "
                                    f"执行加减法编码{execution}时删除了原规则的必要元素，予跳过")
                    if continue_flag:
                        continue
                    if len(opt_mdf['备选元素库包含null']) >= 1:
                        opt_mdf['备选元素库包含null'] = list(opt_mdf['备选元素库包含null'])
                    else:
                        opt_mdf.pop('备选元素库包含null')

                    for trait_group in tp_mdf.keys():
                        tp_mdf[trait_group] = {trait:tp_mdf[trait_group][trait] for trait in tp_mdf[trait_group]
                                               if tp_mdf[trait_group][trait] != 'NA'}

                    new_psc_num_dict = {'方剂编号': copy.deepcopy(row_modif['方剂编号']) + '-' + '&'.join(rule_num_mdf)}

                    new_rule_num_dict = {'规则编号': copy.deepcopy(row_modif['规则编号']) + '-' + '&'.join(rule_num_mdf)}

                    new_cps_dict = copy.deepcopy(row_modif['必要元素库A':'必要元素库J'].to_dict())
                    for column in new_cps_dict.keys():
                        if new_cps_dict[column] == 'null':
                            new_cps_dict[column] = []
                        else:
                            new_cps_dict[column] = new_cps_dict[column].split('；')

                    for new_cps_elems in pdt_cps_mdf:
                        for column in new_cps_dict.keys():
                            if len(new_cps_dict[column]) > 0:
                                for new_cps_elem in new_cps_elems:
                                    if new_cps_elem not in new_cps_dict[column]:
                                        new_cps_dict[column].append(new_cps_elem)


                    new_opt_dict = copy.deepcopy(row_modif['备选元素库A':'备选元素库AD'].to_dict())
                    for column in new_opt_dict.keys():
                        if new_opt_dict[column] == 'null':
                            new_opt_dict[column] = []
                        else:
                            new_opt_dict[column] = new_opt_dict[column].split('；')

                    for new_opt_elems in opt_mdf.values():
                        for column in new_opt_dict.keys():
                            if len(new_opt_dict[column]) > 0:
                                continue
                            else:
                                new_opt_dict[column] = new_opt_elems
                                break

                    for del_elem in del_mdf:
                        for column in new_cps_dict.keys():
                            if del_elem in new_cps_dict[column]:
                                new_cps_dict[column].remove(del_elem)
                        for column in new_opt_dict.keys():
                            if del_elem in new_opt_dict[column]:
                                new_opt_dict[column].remove(del_elem)

                    new_tp_dict = copy.deepcopy(row_modif['舌淡白':'null.10'].to_dict())
                    for trait_group in tp_mdf.keys():
                        for trait in tp_mdf[trait_group]:
                            if tp_mdf[trait_group][trait] == 'F':
                                new_tp_dict[trait] = 'null'
                            else:
                                if new_tp_dict[trait] == 'F':
                                    continue_flag = True
                                    logger.warn(f"方名'{psc_name}' 方剂编号'{psc_num}', "
                                                f"执行加减法编码{execution}舌脉元素库加减法执行时存在矛盾")
                                    break
                                if new_tp_dict[trait] == 'null':
                                    new_tp_dict[trait] = '；；'

                                tp1 = [set(filter(lambda x: x != '', x.split('&')))
                                       for x in new_tp_dict[trait].split('；')]
                                tp2 = [set(filter(lambda x: x != '', x.split('&')))
                                       for x in tp_mdf[trait_group][trait].split('；')]
                                tp2_nes = []
                                for x in tp2[2]:
                                    temp = []
                                    for y in x.split('/'):
                                        if 'F' in new_tp_dict[y]:
                                            continue
                                        temp.append(y)
                                    if temp:
                                        tp2_nes.append('/'.join(temp))
                                    else:
                                        continue_flag = True
                                        logger.warning(f"方名'{psc_name}' 方剂编号'{psc_num}', "
                                                       f"执行加减法编码{execution}舌脉加减法执行时存在矛盾")
                                tp1[0].update(tp2[0])
                                tp1[1].update(tp2[1])
                                tp1[2].update(tp2_nes)

                                new_tp_dict[trait] = '；'.join(['&'.join(xs) for xs in tp1])


                    new_psc_dict = {'方名': copy.deepcopy(row_modif['方名']).split('；')}
                    for adding_psc in psc_mdf:
                        if adding_psc not in new_psc_dict['方名']:
                            new_psc_dict['方名'].append(adding_psc)

                    new_medic_dict = copy.deepcopy(row_modif['药物1':'药物50'].to_dict())
                    for adding_med in medic_mdf['加法']:
                        if adding_med in new_medic_dict.values():
                            continue
                        for num in new_medic_dict.keys():
                            if new_medic_dict[num] != 'null':
                                continue
                            else:
                                new_medic_dict[num] = adding_med
                                break
                    for dropping_med in medic_mdf['减法']:
                        dropping_med = dropping_med.replace('+', '\\+')
                        for num in new_medic_dict.keys():
                            if re.match(dropping_med + '[+]*', new_medic_dict[num]):
                                new_medic_dict[num] = '-'
                    if continue_flag:
                        continue
                    pattern = copy.deepcopy(row_modif)
                    merged_dict_A = {**new_cps_dict, **new_opt_dict, **new_psc_dict}

                    merged_dict_B = {**new_psc_num_dict, **new_rule_num_dict, **new_tp_dict, **new_medic_dict}
                    for colum in merged_dict_A.keys():
                        if len(merged_dict_A[colum]) == 0:
                            pattern[colum] = None
                        else:
                            pattern[colum] = '；'.join(merged_dict_A[colum])
                    for colum in merged_dict_B.keys():
                        pattern[colum] = merged_dict_B[colum]
                    new_df.loc[idx] = pattern
                    idx += 1
        new_df.drop('A类加减法', axis=1, inplace=True)
        new_df.drop('B类加减法', axis=1, inplace=True)
        return new_df

    def _generate_common_indications(self):
        org_df, new_df = copy.deepcopy(self.df), copy.deepcopy(self.df)
        content = dict()
        for i, row in org_df.iterrows():
            if row['方剂编号'] not in content.keys():
                content[row['方剂编号']] = set()
            if row['A类加减法'] == 'null':

                content[row['方剂编号']].update(
                    chain.from_iterable([txt.replace('; ', '；').replace(';', '；').split('；')
                                         for txt in row['必要元素库A':'治疗元素库'].values.tolist()]))
            if 'null' in content[row['方剂编号']]:
                content[row['方剂编号']].remove('null')
        for i, row in new_df.iterrows():
            temp = content[row['方剂编号']]
            all_elems = set(chain.from_iterable([txt.replace('; ', '；').replace(';', '；').split('；')
                                 for txt in row['必要元素库A':'治疗元素库'].values.tolist()]))

            new_df.loc[i]['共享元素库'] = '；'.join(temp - all_elems)
        return new_df

    def generate_rules(self, save_path:str, head_rule:bool=False, share_common_indication:bool=False,
                       apply_modification:bool=True, shuffle:bool=True):
        if head_rule:
            self.df = self._generate_head_rules()
        if share_common_indication:
            self.df = self._generate_common_indications()
        if apply_modification:
            self.df = self._generate_modification()
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.fillna('')
        self.df.replace('null', np.NaN, inplace=True)
        self.df.to_excel(save_path, sheet_name='规则', index=False)


if __name__ == '__main__':
    save_path = os.path.join('root_data', 'root_mod.xlsx')
    rg = RuleGenerator(org_file_path='./root_data/root_change.xlsx')
    rg.generate_rules(save_path)
