#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/5/10 14:57
@file: train.py
@author: baidq
@Software: PyCharm
@desc:
'''

from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam

from nlu.intent_recg_bert.bert_model import build_bert_model
from nlu.intent_recg_bert.data_helper import load_data

#定义超参数和配置文件
class_nums = 13
maxlen = 128
batch_size = 8

config_path='D:\download\Data\Bert Model\chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path='D:\download\Data\Bert Model\chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'D:\download\Data\Bert Model\chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path)

class data_generator(DataGenerator):
    """
    数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):

            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []



def train():
    # 加载数据集
    train_data = load_data('./data/train.csv')
    test_data = load_data('./data/test.csv')

    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)

    model = build_bert_model(config_path, checkpoint_path, class_nums)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(5e-6),
        metrics=['accuracy']
    )

    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=2,
                                              verbose=2,
                                              mode='min'
                                              )
    # 保存最好的模型
    best_model_filepath = 'best_model_weights'
    checkpoint = keras.callbacks.ModelCheckpoint(best_model_filepath,
                                                 monitor='val_loss',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 mode = 'min')

    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=10,
                        validation_data=test_generator.forfit(),
                        validation_steps=len(test_generator),
                        shuffle=True,
                        callbacks=[earlystop, checkpoint])

    # 评估
    model.load_weights('best_model_weights')

    test_pred = []
    test_true = []
    for x, y in test_generator:
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    test_true = test_data[:, 1].tolist()
    print(set(test_true))
    print(set(test_pred))

    target_names = [line.strip() for line in open('label', 'r', encoding='utf8')]
    print(classification_report(test_true, test_pred, target_names=target_names))



if __name__ == '__main__':
    train()