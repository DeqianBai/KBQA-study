#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/5/7 16:30
@file: app.py
@author: baidq
@Software: PyCharm
@desc:
'''
import json
import flask
import pickle
import ahocorasick
import numpy as np
from gevent import pywsgi
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.sequence import pad_sequences

from crf_layer import CRF
from bilstm_crf_model import BiLstmCrfModel

class NerBaseDict(object):
    """
    用Aho-Corasick算法匹配疾病名称实体
    AC自动机：字符串多模匹配算法
    参考：
        https://benarvintec.com/2018/11/26/%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0%E4%B9%8BAho-Corasick/
        https://zhuanlan.zhihu.com/p/158767004
        https://pyahocorasick.readthedocs.io/en/latest/
    """
    def __init__(self, dict_path):
        super(NerBaseDict, self).__init__()
        self.dict_path = dict_path
        self.region_words = self.load_dict(self.dict_path)
        self.region_tree = self.build_actree(self.region_words)

    def load_dict(self,path):
        """
        下载疾病名称字典
        :return: 包含疾病名称的list
        """
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)

    def build_actree(self, wordlist):
        """
        构建Tire Tree
        :param wordlist: 包含疾病名称的list
        :return:  一个Aho-Corasick自动机
        """
        actree = ahocorasick.Automaton()            # 创建自动机，将Automaton类用作Trie。
        for index, word in enumerate(wordlist):     # 将(疾病名称索引，疾病名称)的元组作为值添加到添加到Trie的每个键字符串中
            actree.add_word(word, (index, word))
        actree.make_automaton()                     # 将Trie转换为Aho-Corasick自动机以启用Aho-Corasick搜索
        return actree

    def recognize(self, text):
        """
        疾病名称实体匹配
        :param text:
        :return:
        """
        item = {"string": text, "entities": []}

        region_wds = []
        for i in self.region_tree.iter(text):   # 所有子串匹配iter: 在输入字符串中搜索所有出现的键
            wd = i[1][1]                        # i: (6, (3228, '心脏病'))
            region_wds.append(wd)               # region_wds: ['心脏病']

        stop_wds = []
        for wd1 in region_wds:
            for wd2 in region_wds:
                if wd1 in wd2 and wd1!= wd2:
                    stop_wds.append(wd1)
        final_wds = [i for i in region_wds if i not in stop_wds]
        item["entities"] = [{"word":i, "type":"disease", "recog_label":"dict"} for i in final_wds]
        # item: {'string': '请问得了心脏病怎么办呢', 'entities': [{'word': '心脏病', 'type': 'disease', 'recog_label': 'dict'}]}
        return item


class MedicalNerModel(object):
    """
    用于医疗领域的命名实体识别模型
    基于bilstm-crf实现
    """
    def __init__(self):
        super(MedicalNerModel, self).__init__()
        self.word2id,_,self.id2tag = pickle.load(
                open("./checkpoint/word_tag_id.pkl","rb")
            )
        self.model = BiLstmCrfModel(80,2410,200,128,24).build()
        self.model.load_weights('./checkpoint/best_bilstm_crf_model.h5')

        self.nbd = NerBaseDict('./checkpoint/diseases.json')

    def tag_parser(self, string, tags):
        """
        对tag进行反向解码，将tag组成实体后进行输出
        :param string:
        :param tags:
        :return:
        """
        item = {"string": string, "entities": []}
        entity_name = ""
        flag = []
        visit = False
        for char, tag in zip(string, tags):
            if tag[0] == "B":
                if entity_name != "":
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    item["entities"].append({"word": entity_name, "type": y[0]})
                    flag.clear()
                    entity_name = ""
                entity_name += char
                flag.append(tag[2:])

            elif tag[0] == "I":
                entity_name += char
                flag.append(tag[2:])
            else:
                if entity_name != "":
                    x = dict((a, flag.count(a)) for a in flag)
                    y = [k for k, v in x.items() if max(x.values()) == v]
                    item["entities"].append({"word": entity_name, "type": y[0]})
                    flag.clear()
                flag.clear()
                entity_name = ""

        if entity_name != "":
            x = dict((a, flag.count(a)) for a in flag)
            y = [k for k, v in x.items() if max(x.values()) == v]
            item["entities"].append({"word": entity_name, "type": y[0]})

        return item


    def predict(self,texts):
        """
        texts 为一维列表，元素为字符串
        texts = ["淋球菌性尿道炎的症状","上消化道出血的常见病与鉴别"]
        """
        max_len = 80
        X = [[self.word2id.get(word,1) for word in list(x)] for x in texts ]
        X = pad_sequences(X,maxlen=max_len,value=0)
        pred_id = self.model.predict(X)
        res = []
        for text,pred in zip(texts,pred_id):
            tags = np.argmax(pred,axis=1)
            tags = [self.id2tag[i] for i in tags if i!=0]
            # self.tag_parser(text,tags) = {'string': '请问得了心脏病怎么办呢', 'entities': [{'word': '心脏病', 'type': 'disease'}]}
            res.append(self.tag_parser(text,tags))

        for text in texts:                      # texts: ['请问得了心脏病怎么办呢']
            entities = self.nbd.recognize(text) # entities: {'string': '请问得了心脏病怎么办呢', 'entities': [{'word': '心脏病', 'type': 'disease', 'recog_label': 'dict'}]}
            if entities["entities"]:
                res.append(entities)
        return res


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
graph = tf.get_default_graph()
set_session(sess)

NER = MedicalNerModel()
app = flask.Flask(__name__)
@app.route("/service/api/medical_ner",methods=["GET","POST"])
def medical_ner():
    data = {"success":0}
    result = []
    text_list = flask.request.get_json()["text_list"]
    with graph.as_default():
        set_session(sess)
        result = NER.predict(text_list)

    data["data"] = result
    data["success"] = 1

    return flask.jsonify(data)

if __name__ == '__main__':
    server = pywsgi.WSGIServer(("0.0.0.0",60061), app)
    server.serve_forever()
    # ner = MedicalNerModel()
    # r = ner.predict(["淋球菌性尿道炎的症状","上消化道出血的常见病与鉴别"])
    # print(r)