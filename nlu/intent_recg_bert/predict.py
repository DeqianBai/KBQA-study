#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/5/17 11:26
@file: predict.py
@author: baidq
@Software: PyCharm
@desc:
'''
from bert4keras.snippets import sequence_padding, DataGenerator
from nlu.intent_recg_bert.data_helper import load_data
from bert4keras.tokenizers import Tokenizer
from nlu.intent_recg_bert.bert_model import build_bert_model

maxlen = 128
batch_size = 8
config_path = 'D:\download\Data\Bert Model\chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:\download\Data\Bert Model\chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'D:\download\Data\Bert Model\chinese_L-12_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(dict_path)
class_nums = 13


class data_generator(DataGenerator):
    """
    数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            print(text)
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            print(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def predict_batch():
    """
    以batch的形式进行预测
    :return:
    """
    test_data = load_data('test_data.csv')
    test_generator = data_generator(test_data, batch_size)

    model = build_bert_model(config_path, checkpoint_path, class_nums)
    model.load_weights('./model/best_model_weights')

    test_pred = []
    for x, y in test_generator:
        print(x)
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    return test_pred


def predict(text):
    """
    输入单个文本进行预测
    :param text:
    :return:
    """
    model = build_bert_model(config_path, checkpoint_path, class_nums)
    model.load_weights('./model/best_model_weights')
    print(text)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    print(token_ids)
    print(segment_ids)
    # token_ids = sequence_padding(token_ids)
    # segment_ids= sequence_padding(segment_ids)

    predict_result = model.predict([[token_ids], [segment_ids]]).argmax(axis=1)

    return predict_result
if __name__ == '__main__':
    x = "你好！茶碱缓释片有什么要注意的吗？哪些人是不适用或要慎用的？"
    x1 = "最近舌尖上有疮，用盐治疗过，效果不大。我可不可以用牛黄解毒片治？如果可以，要怎么治？"
    x2 = "男性小便疼痛吃什么药"
    x3 ="我患有精神分裂症吃利培酮能同时吃宁神颗粒和脑灵素胶囊吗"

    result = predict(x)
    print(result)

    # result = predict_batch()
    # print(result)