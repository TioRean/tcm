import copy

from pymysql import *
import json
import pandas as pd
from itertools import chain
import os
import re
import warnings
from typing import Union

class SQLAdministrator:
    def __init__(self, name):
        self.name = name
        self.root = RootTable(name)

    def init_sql(self):
        self.root.mk_table()

    def save_root_data(self, TCMRule_path:str, vocab_path:str):
        root_elements_list = self.root._get_root_elems_list(TCMRule_path)
        contradictory = self.root._get_contradictory_vocab(vocab_path)
        accessory = self.root._get_accessory_vocab(vocab_path)
        noises = self.root._get_noises_dict(vocab_path)
        modularized = self.root._get_modularized_vocab(vocab_path)
        synonym = self.root._get_synonym_vocab(vocab_path)
        self.root.insert(root_elements_list, synonym, accessory, contradictory, noises, modularized)

class SQL:
    conn_params = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'charset': 'utf8',
        'passwd': '456258',
        'db': 'wanglu'}

    def __init__(self, conn_params):
        self.__host = conn_params['host']
        self.__port = conn_params['port']
        self.__db = conn_params['db']
        self.__user = conn_params['user']
        self.__passwd = conn_params['passwd']
        self.__charset = conn_params['charset']

    def __connect(self):
        self.__conn = connect(host=self.__host,
                              port=self.__port,
                              db=self.__db,
                              user=self.__user,
                              passwd=self.__passwd,
                              charset=self.__charset)
        self.__cursor = self.__conn.cursor()

    def __close(self):
        self.__cursor.close()
        self.__conn.close()

    def _edit(self, sql, params=None):
        count = 0
        try:
            self.__connect()
            if params is not None:
                count = self.__cursor.execute(sql, params)
            else:
                count = self.__cursor.execute(sql)
            self.__conn.commit()
            self.__close()
        except Exception as e:
            print(e)
        return count

    def _mk_table(self, sql):
        try:
            self.__connect()
            self.__cursor.execute(sql)
            self.__close()
        except Exception as e:
            print(e)

    def _get_one(self, sql, params=None):
        result = None
        try:
            self.__connect()
            if params is not None:
                self.__cursor.execute(sql, params)
            else:
                self.__cursor.execute(sql)
            result = self.__cursor.fetchone()
            self.__close()
        except Exception as e:
            print(e)
        return result

    def _get_all(self, sql, params=None):
        li = ()
        try:
            self.__connect()
            if params is not None:
                self.__cursor.execute(sql, params)
            else:
                self.__cursor.execute(sql)
            li = self.__cursor.fetchall()
            self.__close()
        except Exception as e:
            print(e)
        return li

    def _insert(self, sql, params):
        return self._edit(sql, params)

    def _update(self, sql, params):
        return self._edit(sql, params)

class RootTable(SQL):
    def __init__(self, name):
        super(RootTable, self).__init__(self.conn_params)
        self.name = name
        self.cls_elems_address = self.name + '_cls_elems'
        self.explainer_address = self.name + '_explainer'

    def mk_table(self):
        sql = '''
                            CREATE TABLE Root(
                            `id` INT NOT NULL AUTO_INCREMENT COMMENT 'id',
                            `name` varchar(100) DEFAULT NULL COMMENT '名称',
                            `root_elems_list` json DEFAULT NULL COMMENT '中医训练知识库',
                            `synonym` json DEFAULT NULL COMMENT '同义词元素库',
                            `accessory` json DEFAULT NULL COMMENT '附属元素库',
                            `contradictory` json DEFAULT NULL COMMENT '矛盾元素库',
                            `noises` json DEFAULT NULL COMMENT '噪声元素库',
                            `modularized` json DEFAULT NULL COMMENT '模块化词表库',
                            `cls_elems_address` varchar(100) DEFAULT NULL COMMENT '类别-元素词表地址',
                            `explainer_address` varchar(100) DEFAULT NULL COMMENT '解释知识库地址',
                            `update_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '更新时间',
                            `note` varchar(2000) DEFAULT NULL COMMENT '注释',
                            PRIMARY KEY (`id`),
                            UNIQUE KEY unique_name (name) 
                            ) ENGINE=InnoDB;
                        '''
        self._mk_table(sql)

    def get_rule_element_list(self):
        sql = "select root_elems_list from Root where name=%s"
        result = json.loads(self._get_one(sql, (self.name))[0])
        return result

    def get_accessory(self):
        sql = "select accessory from Root where name=%s"
        result  = json.loads(self._get_one(sql, (self.name))[0])
        return result

    def get_contradictory(self):
        sql = "select contradictory from Root where name=%s"
        result  = json.loads(self._get_one(sql, (self.name))[0])
        return result

    def get_noises(self):
        sql = "select noises from Root where name=%s"
        result = json.loads(self._get_one(sql, (self.name))[0])
        return result

    def get_modularized(self):
        sql = "select modularized from Root where name=%s"
        modularized, tong_body, tong_coating, pul = json.loads(self._get_one(sql, (self.name))[0])
        cls_module_paths = {cls_module[1]: cls_module[0] for cls_module in modularized.values()}
        tong_body = {k: set(v) for k, v in tong_body.items()}
        tong_coating = {k: set(v) for k, v in tong_coating.items()}
        pul = {k: set(v) for k, v in pul.items()}
        return modularized, cls_module_paths, tong_body, tong_coating, pul

    def get_synonym(self):
        sql = "select synonym from Root where name=%s"
        result = json.loads(self._get_one(sql, (self.name))[0])
        return result

    def get_cls_elems_address(self):
        sql = "select cls_elems_address from Root where name=%s"
        result = json.loads(self._get_one(sql, (self.name))[0])
        return result

    def get_explainer_address(self):
        sql = "select explainer_address from Root where name=%s"
        result = json.loads(self._get_one(sql, (self.name))[0])
        return result

    def get_all(self):
        sql = "select * from Root where name=%s"
        result = self._get_all(sql, (self.name))
        return result

    def insert(self, root_elems_list, synonym, accessory, contradictory, noises, modularized):
        sql = "insert into Root (name, root_elems_list, synonym, accessory, contradictory, noises, modularized, cls_elems_address, " \
              "explainer_address) values (%s,%s,%s,%s,%s,%s,%s,%s,%s) " \
              "ON DUPLICATE KEY UPDATE root_elems_list=%s, synonym=%s, accessory=%s, contradictory=%s, noises=%s, modularized=%s"
        root_elems_list = json.dumps(root_elems_list, ensure_ascii=False)
        synonym = json.dumps(synonym, ensure_ascii=False)
        accessory = json.dumps(accessory, ensure_ascii=False)
        contradictory = json.dumps(contradictory, ensure_ascii=False)
        noises = json.dumps(noises, ensure_ascii=False)
        modularized = json.dumps(modularized, ensure_ascii=False)
        return self._insert(sql, (self.name, root_elems_list, synonym, accessory, contradictory, noises, modularized,
                                  self.cls_elems_address, self.explainer_address, root_elems_list, synonym, accessory,
                                  contradictory, noises, modularized))

    def update(self, trg_column, trg_data):
        if trg_column in ('root_elems_list', 'accessory', 'contradictory', 'noises'):
            trg_data = json.dumps(trg_data, ensure_ascii=False)
        sql = "update Root set %s=%s where name=%s"
        return self._update(sql, (trg_column, trg_data, self.name))

    @staticmethod
    def _get_root_elems_list(TCMRule_path:str):
        def get_elements(dataframe_loc)->list[Union[str, list, set, dict]]:
            if isinstance(dataframe_loc, str):
                if dataframe_loc == 'None':
                    return []
                else:
                    return dataframe_loc.replace('； ', '；').replace('; ', '；').replace(';', '；').split('；')
            else:
                return [x.replace('； ', '；').replace('; ', '；').replace(';', '；')
                        for x in filter(lambda x: x != 'None', dataframe_loc)]

        def tongue_pulse_add(arr: list):
            if arr == []:
                return ['null；；']
            else:
                return arr
        def get_elements_list_for_tongue_pulse(dataframe, i):
            trait_groups = (
                ('舌淡白', '舌青'), ('舌老', 'null'), ('白苔', '黑苔'), ('苔少', '苔厚'), ('苔水滑', 'null.1'),
                ('苔腻', 'null.2'), ('脉数', 'null.3'), ('脉促', 'null.4'), ('脉浮', 'null.5'), ('脉大', 'null.6'),
                ('脉虚', 'null.7'), ('脉滑', 'null.8'), ('脉弦', 'null.9'), ('革脉', 'null.10'))
            content = []
            for trait_group in trait_groups:
                traits = get_elements(dataframe.loc[i, trait_group[0]:trait_group[1]])
                if traits == []:
                    content.append(['null；；'])
                elif 'F' in traits:
                    content.append(['F；；'])
                else:
                    content.append(traits)
            return content

        def process_elems_with_special_tokens(elem_li):
            elem_dict = dict()
            for elem in elem_li:
                elem = elem.replace(' ', '').replace('《', '<').replace('》', '>').strip(',，;；')
                if re.search(r'<[cC]>', elem):
                    denote = re.search(r'<[cC]>', elem).group()
                    if denote not in elem_dict.keys():
                        elem_dict[denote] = set()
                    elem_dict[denote].add(elem.replace(denote, ''))
                elif re.search(r'<[oO][a-zA-Z]*>', elem):
                    denote = re.search(r'<[oO][a-zA-Z]*>', elem).group()
                    if denote not in elem_dict.keys():
                        elem_dict[denote] = set()
                    elem_dict[denote].add(elem.replace(denote, ''))
                else:
                    if '<own>' not in elem_dict.keys():
                        elem_dict['<own>'] = set()
                    elem_dict['<own>'].add(elem)
            elem_dict = {k: list(filter(lambda x: x != '' and x != 'null', v)) for k, v in elem_dict.items()}
            elem_dict = {k: v for k, v in elem_dict.items() if len(v) > 0}
            return elem_dict

        df = pd.read_excel(TCMRule_path)
        df = df.fillna('None')
        df_1 = df.loc[:, '方剂编号':'共享元素库'].astype(str)
        df_2 = df.loc[:, '舌淡白':'null.10'].astype(str)
        df_3 = df.loc[:, '方名':'药物50'].astype(str)
        for column in list(df_2.columns):
            if re.match(r'null\.\d+', column):
                temp = 'null'
                df_2.loc[:, column] = df_2[column].str.replace('T', temp, regex=False)
            else:
                df_2.loc[:, column] = df_2[column].str.replace('T', column, regex=False)
        root_elements_list = []
        for i in range(0, len(df)):
            prescription_num = get_elements(df_1.loc[i, '方剂编号'])
            prescription_num = str(prescription_num).replace('[', '').replace(']', '').replace("'", '')

            rule_num = get_elements(df_1.loc[i, '规则编号'])
            rule_num = str(rule_num).replace('[', '').replace(']', '').replace("'", '')

            classification = get_elements(df_1.loc[i, '类别'])
            classification = str(classification).replace('[', '').replace(']', '').replace("'", '').replace(', ', '&')

            compulsories = get_elements(df_1.loc[i, '必要元素库A':'必要元素库J'])
            compulsories = [compulsory.split('；') for compulsory in compulsories]
            while len(compulsories) < 10:
                compulsories.append([])

            optionals:list[Union[str, list, dict]] = get_elements(df_1.loc[i, '备选元素库A':'备选元素库AD'])
            optionals = [optional.split('；') for optional in optionals]
            while len(optionals) < 30:
                optionals.append([])
            shared = get_elements(df_1.loc[i, '共享元素库'])
            optionals.append(shared)
            tongue_pulse = get_elements_list_for_tongue_pulse(dataframe=df_2, i=i)
            prescription = get_elements(df_3.loc[i, '方名'])
            medicines = get_elements(df_3.loc[i, '药物1':'药物50'])

            root_elements_list.append(
                [prescription_num, rule_num, classification, compulsories, optionals, tongue_pulse, prescription, medicines])

        return root_elements_list

    @staticmethod
    def _process_root_elems_list(root_elems_list: list):
        result = copy.deepcopy(root_elems_list)
        for i, root_elems in enumerate(root_elems_list):
            _, _, cls, compulsory, optionals, tongue_pulse, _, _ = root_elems
            tongue_pulse = [[trait.replace('; ', '；').split('；')[0] for trait in trait_li]
                            for trait_li in tongue_pulse]
            result[i][7] = tongue_pulse
        return result

    @staticmethod
    def _get_noises_dict(vocab_path:str):
        df = pd.read_excel(vocab_path, sheet_name='noises', skiprows=0)
        df = df.fillna('null')
        inducement_vocab = list(set(filter(lambda x: x != 'null', list(df['诱因']))))
        sym_phy_vocab = list(set(filter(lambda x: x != 'null', list(df['症状体征']))))
        tong_pul_vocab = list(set(filter(lambda x: x != 'null', list(df['舌脉']))))
        keys = sym_phy_vocab + tong_pul_vocab
        values = ['sym_phy'] * len(sym_phy_vocab) + ['tong_pul'] * len(tong_pul_vocab)
        content = {k: v for k, v in zip(keys, values)}
        return content

    @staticmethod
    def _get_modularized_vocab(vocab_path: str):
        df = pd.read_excel(vocab_path, sheet_name='modularized_vocab')
        df = df.fillna('null')
        content = dict()
        tong_body = dict()
        tong_coating = dict()
        pul = dict()
        for i, row in df.iterrows():
            if row['大类别'] == '舌质':
                if row['类别'] not in tong_body:
                    tong_body[row['类别']] = set()
                tong_body[row['类别']].add(row['元素'])
                continue
            if row['大类别'] == '舌苔':
                if row['类别'] not in tong_coating:
                    tong_coating[row['类别']] = set()
                tong_coating[row['类别']].add(row['元素'])
                continue
            if row['大类别'] == '脉象':
                if row['类别'] not in pul:
                    pul[row['类别']] = set()
                pul[row['类别']].add(row['元素'])
                continue
            content[row['元素']] = [row['大类别'], row['类别']]
        tong_body = {k: list(v) for k, v in tong_body.items()}
        tong_coating = {k: list(v) for k, v in tong_coating.items()}
        pul = {k: list(v) for k, v in pul.items()}
        return content, tong_body, tong_coating, pul

    @staticmethod
    def _get_synonym_vocab(vocab_path: str):
        df = pd.read_excel(vocab_path, sheet_name='modularized_vocab', skiprows=0)
        df = df.fillna('None')
        content = dict()
        for i, row in df.iterrows():
            if row['元素'] not in content.keys():
                content[row['元素']] = set()
                content[row['元素']].add(row['元素'])
            if row['同义词'] != 'None':
                content[row['元素']].update(row['同义词'].replace('; ', '；').replace(';', '；').split('；'))
        content = {k: list(v) for k, v in content.items()}
        return content

    @staticmethod
    def _get_accessory_vocab(vocab_path: str):
        df = pd.read_excel(vocab_path, sheet_name='accessory', skiprows=0)
        df = df.fillna('None')
        content = dict()
        for i, row in df.iterrows():
            if row['key'] not in content.keys():
                content[row['key']] = set()
            content[row['key']].update(row['value'].replace('; ', '；').replace(';', '；').split('；'))
        content = {k: list(v) for k, v in content.items()}
        return content

    @staticmethod
    def _get_contradictory_vocab(vocab_path: str):
        df = pd.read_excel(vocab_path, sheet_name='contradictory', skiprows=0)
        df = df.fillna('None')
        content = dict()
        for i, row in df.iterrows():
            if row['key'] not in content.keys():
                content[row['key']] = set()
            content[row['key']].update(row['value'].replace('; ', '；').replace(';', '；').split('；'))
        content = {k: list(v) for k, v in content.items()}
        return content
