#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/5/8 10:01
@file: build_baiduie_data.py
@author: baidq
@Software: PyCharm
@desc:
'''

import sys
import json
from tqdm import tqdm
import codecs
import numpy as np
import pandas as pd

# raw_data_dir, training_data_dir = sys.argv[1], sys.argv[2]
raw_data_dir = './data/'
training_data_dir = 'processed_data'

RANDOM_SEED = 2019

rel_set = set()

text_len = []

# 处理训练数据
train_data = []
i = 0
with open(raw_data_dir + '/train_data/train_data.json', encoding='utf8') as f:
    for l in tqdm(f.readlines()):
        a = json.loads(l)
        # 打印第一条原始数据进行查看
        if i == 0:
            print(json.dumps(a, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
        if not a['spo_list']:
            continue

        triple_list = []
        for spo in a['spo_list']:
            s = spo['subject']
            p = spo['predicate']
            o_dict = spo['object']
            for k in o_dict.keys():
                triple_list.append((s, p + '_' + k, o_dict[k]))
                rel_set.add(p + '_' + k)

        line = {
            'text': a['text'],
            'triple_list': triple_list
        }
        # 打印第一条处理后的数据进行查看
        if i == 0:
            print(json.dumps(line, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
        train_data.append(line)
        text_len.append((len(a['text'])))
        i += 1

df = pd.DataFrame({"text_len": text_len})
print("训练集文本长度统计：\n")
print(df["text_len"].describe())

id2rel = {i: j for i, j in enumerate(sorted(rel_set))}
rel2id = {j: i for i, j in id2rel.items()}

# 保存训练集
with codecs.open(training_data_dir + '/rel2id.json', 'w', encoding='utf-8') as f:
    json.dump([id2rel, rel2id], f, indent=4, ensure_ascii=False)

with codecs.open(training_data_dir + '/train_triples.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


# 处理验证数据
dev_data = []
with open(raw_data_dir + '/dev_data/dev_data.json', encoding='utf8') as f:
    for l in tqdm(f.readlines()):
        a = json.loads(l)
        if not a['spo_list']:
            continue

        triple_list = []
        for spo in a['spo_list']:
            s = spo['subject']
            p = spo['predicate']
            o_dict = spo['object']
            for k in o_dict.keys():
                triple_list.append((s,p+'_'+k,o_dict[k]))
                rel_set.add(p+'_'+k)

        line = {
                'text': a['text'],
                'triple_list': triple_list
               }
        dev_data.append(line)


dev_len = len(dev_data)
random_order = list(range(dev_len))
np.random.seed(RANDOM_SEED)
np.random.shuffle(random_order)

# 取验证集的前50%作为测试集，余下的作为验证集
test_data = [dev_data[i] for i in random_order[:int(0.5 * dev_len)]]
dev_data = [dev_data[i] for i in random_order[int(0.5 * dev_len):]]

# 保存验证集和测试集
with codecs.open(training_data_dir+'/dev_triples.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)

with codecs.open(training_data_dir+'/test_triples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)