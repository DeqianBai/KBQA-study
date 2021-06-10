#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/6/4 9:34
@file: clf_model.py
@author: baidq
@Software: PyCharm
@desc:
'''
import os
import pickle
import numpy as np
from sklearn import svm

class CLFModel(object):
    """
    闲聊意图分类器
    """
    def __init__(self, model_save_path):
        super(CLFModel, self).__init__()
        self.model_save_path = model_save_path
        self.id2label = pickle.load(open(os.path.join(self.model_save_path, "id2label.pkl"), "rb"))
        self.vec = pickle.load(open(os.path.join(self.model_save_path, "vec.pkl"), "rb"))
        self.LR_clf = pickle.load(open(os.path.join(self.model_save_path, "LR.pkl"), "rb"))
        self.GBDT_clf = pickle.load(open(os.path.join(self.model_save_path, "gbdt.pkl"),"rb"))

    def predict(self, text):
        """
        模型融合预测
        :param text:
        :return:
        """
        text = ' '.join(list(text.lower()))
        text = self.vec.transform([text])
        predict_1 = self.LR_clf.predict_proba(text)
        predict_2 = self.GBDT_clf.predict_proba(text)
        label = np.argmax((predict_1+predict_2)/2, axis=1)
        return self.id2label.get(label[0])

if __name__ == '__main__':
    model = CLFModel("./model_file/")

    text = "你是谁"
    text1 = "不聊了"
    label = model.predict(text1)
    print(label)

