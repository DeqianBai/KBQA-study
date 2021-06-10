#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/6/4 16:10
@file: app.py
@author: baidq
@Software: PyCharm
@desc:
'''
import flask
import tensorflow as tf
from gevent import pywsgi
from bert4keras.tokenizers import Tokenizer
from keras.backend.tensorflow_backend import set_session
from bert_model import build_bert_model


class BertIntentModel(object):
    """
    基于bert实现的医疗意图识别模型
    """
    def __init__(self):
        super(BertIntentModel, self).__init__()
        self.dict_path = 'D:\download\Data\Bert Model\chinese_L-12_H-768_A-12/vocab.txt'
        self.config_path = 'D:\download\Data\Bert Model\chinese_L-12_H-768_A-12/bert_config.json'
        self.checkpoint_path = 'D:\download\Data\Bert Model\chinese_L-12_H-768_A-12/bert_model.ckpt'

        self.label_list  = [line.strip() for line in open("label", "r", encoding="utf8")]
        self.id2label = {idx:label for idx, label in enumerate(self.label_list)}

        self.tokenizer = Tokenizer(self.dict_path)
        self.model = build_bert_model(self.config_path, self.checkpoint_path, 13)
        self.model.load_weights("./model/best_model_weights")

    def predict(self, text):
        """
        对用户输入的单条文本进行预测
        :param text:
        :return:
        """
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=60)
        predict = self.model.predict([[token_ids], [segment_ids]])
        rst = {l:p for l,p in zip(self.label_list, predict[0])}
        rst = sorted(rst.items(), key= lambda kv:kv[1], reverse=True)
        print(rst[0])
        intent, confidence = rst[0]
        return {"intent": intent, "confidence":float(confidence)}


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
graph = tf.get_default_graph()
set_session(sess)

# 医疗意图识别分类器
BIM= BertIntentModel()
app = flask.Flask(__name__)
@app.route("/service/api/bert_intent_recognize", methods=["GET", "POST"])
def bert_intent_recognize():
    data = {"success":0}
    param = flask.request.get_json()
    print("param:", param)
    text = param["text"]
    with graph.as_default():
        set_session(sess)
        result = BIM.predict(text)
    data["data"] = result
    data["success"] = 1

    return flask.jsonify(data)

if __name__ == '__main__':
    sever = pywsgi.WSGIServer(("0.0.0.0", 60062), app)
    sever.serve_forever()
    # r = BIM.predict("淋球菌性尿道炎的症状")
    # print(r)

