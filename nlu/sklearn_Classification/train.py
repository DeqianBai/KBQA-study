#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/6/4 9:54
@file: train.py
@author: baidq
@Software: PyCharm
@desc:
'''
import os
import pickle
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

seed = 222
random.seed(seed)
np.random.seed(seed)

def load_data(data_path):
    """
    加载并处理intent_recog_data.txt文件中的训练数据
    :param data_path:
    :return:
    """
    X, y = [], []
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            text, label =  line.strip().split(',') # 早上好呀,greet
            text = ' '.join(list(text.lower()))
            X.append(text)
            y.append(label)

    index = np.arange(len(X))
    np.random.shuffle(index)
    X = [X[i] for i in index]
    y = [y[i] for i in index]
    return X, y

def run(data_path, model_save_path):
    """
    进行闲聊意图识别的训练
    :param data_path:
    :param model_save_path:
    :return:
    """
    X, y = load_data(data_path)

    label_set  = sorted(list(set(y)))
    label2id = {label:idx for idx, label in enumerate(label_set)}
    id2label = {idx:label for label, idx in label2id.items()}

    y = [label2id[i] for i in y]

    label_names = sorted(label2id.items(), key=lambda kv:kv[1], reverse=False)
    target_names = [i[0] for i in label_names]
    labels = [i[1] for i in label_names]

    train_X, test_X, train_y, test_y = train_test_split(X, y,   test_size=0.15, random_state=42)

    # 向量化
    vec = TfidfVectorizer(ngram_range=(1,3),
                          min_df=0,
                          max_df=0.9,
                          analyzer="char",
                          use_idf=1,
                          smooth_idf=1,
                          sublinear_tf=1)
    train_X = vec.fit_transform(train_X)
    test_X = vec.transform(test_X)

    #---------------LR---------------
    LR = LogisticRegression(C=8, dual=False, n_jobs=4,
                            max_iter=400,
                            multi_class='ovr',
                            random_state=122)

    LR.fit(train_X, train_y)
    pred  = LR.predict(test_X)
    print('-----------------LR Report--------------')
    print(classification_report(test_y, pred, target_names=target_names))
    print(confusion_matrix(test_y, pred, labels=labels))

    # ---------------GBDT---------------
    GBDT = GradientBoostingClassifier(n_estimators=450,
                                      learning_rate=0.01,
                                      max_depth=8,
                                      random_state=24)

    GBDT.fit(train_X, train_y)
    pred = GBDT.predict(test_X)
    print('-----------------GBDT Report--------------')
    print(classification_report(test_y, pred,target_names=target_names))
    print(confusion_matrix(test_y, pred, labels=labels))
    # -------------融合--------------
    pred_prob1 = LR.predict_proba(test_X)
    pred_prob2 = GBDT.predict_proba(test_X)

    pred = np.argmax((pred_prob1+pred_prob2)/2, axis=1)
    print('-----------------融合 Report--------------')
    print(classification_report(test_y, pred, target_names=target_names))
    print(confusion_matrix(test_y, pred, labels=labels))

    pickle.dump(id2label, open(os.path.join(model_save_path,'id2label.pkl'),'wb'))
    pickle.dump(vec, open(os.path.join(model_save_path,'vec.pkl'),'wb'))
    pickle.dump(LR, open(os.path.join(model_save_path, 'LR.pkl'), 'wb'))
    pickle.dump(GBDT, open(os.path.join(model_save_path, 'gbdt.pkl'), 'wb'))

if __name__ == '__main__':
    run("./data/intent_recog_data.txt", "./model_file/")

