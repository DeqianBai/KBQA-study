#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/5/8 16:55
@file: bert_model.py
@author: baidq
@Software: PyCharm
@desc:
'''

from bert4keras.backend import keras,set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam

set_gelu('tanh')
def textcnn(inputs, kernel_initializer):
    """
    基于keras实现的textcnn
    :param inputs:
    :param kernel_initializer:
    :return:
    """
    # 3,4,5
    cnn1 = keras.layers.Conv1D(256,
                               3,
                               strides=1,
                               padding='same',
                               activation='relu',
                               kernel_initializer=kernel_initializer)(inputs) # shape=[batch_size,maxlen-2,256]
    cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)

    cnn2 = keras.layers.Conv1D(256,
                               4,
                               strides=1,
                               padding='same',
                               activation='relu',
                               kernel_initializer=kernel_initializer)(inputs)
    cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

    cnn3 = keras.layers.Conv1D(256,
                               5,
                               strides=1,
                               padding='same',
                               kernel_initializer=kernel_initializer)(inputs)
    cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

    output = keras.layers.concatenate([cnn1, cnn2, cnn3],axis=-1)
    output = keras.layers.Dropout(0.2)(output)

    return output


def build_bert_model(config_path, checkpoint_path, class_nums):
    """
    构建bert模型用来进行医疗意图的识别
    :param config_path:
    :param checkpoint_path:
    :param class_nums:
    :return:
    """
    # 预加载bert模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        return_keras_model=False
    )

    # 抽取cls 这个token
    cls_features = keras.layers.Lambda(
        lambda x: x[:,0], # 所有行的第一列
        name='cls-token')(bert.model.output) #shape=[batch_size,768]
    # 抽取所有的token，从第二个到倒数第二个
    all_token_embedding = keras.layers.Lambda(
        lambda x: x[:,1:-1],
        name='all-token')(bert.model.output) #shape=[batch_size,maxlen-2,768]

    cnn_features = textcnn(all_token_embedding, bert.initializer) #shape=[batch_size,cnn_output_dim]

    # 特征拼接
    concat_features = keras.layers.concatenate([cls_features, cnn_features], axis=-1)

    dense = keras.layers.Dense(units=512,
                               activation='relu',
                               kernel_initializer=bert.initializer)(concat_features)

    output = keras.layers.Dense(units=class_nums,
                                activation='softmax',
                                kernel_initializer=bert.initializer)(dense)

    model = keras.models.Model(bert.model.input, output)
    print(model.summary())

    return model

if __name__ == '__main__':
    config_path = 'E:/bert_weight_files/bert_wwm/bert_config.json'
    checkpoint_path = 'E:/bert_weight_files/bert_wwm/bert_model.ckpt'
    class_nums = 13
    build_bert_model(config_path, checkpoint_path, class_nums)

