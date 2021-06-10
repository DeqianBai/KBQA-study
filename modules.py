#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/6/4 9:31
@file: modules.py
@author: baidq
@Software: PyCharm
@desc:
'''

import json
import requests
import random
from py2neo import Graph

from nlu.sklearn_Classification.clf_model import CLFModel
from utils.json_utils import dump_user_dialogue_context, load_user_dialogue_context
from config import *

# graph = Graph(host="127.0.0.1",
#             http_port=7474,
#             user="neo4j",
#             password="123456")
graph = Graph("http:localhost:7474", auth=("neo4j","123456"))
# g = Graph(host="localhost", port="7474", auth=("neo4j","123456"))
# host也可以是服务器的地址
# 例如 Graph(host="192.168.1.111", port="7474", auth=("neo4j","test"))也可以访问

clf_model = CLFModel('./nlu/sklearn_Classification/model_file/')

def intent_classifier(text):
    """
    通过post方式请求医疗意图识别分类服务
    基于bert+textcnn实现
    :param text:
    :return:
    """
    url = "http://127.0.0.1:60062/service/api/bert_intent_recognize"
    data = {"text":text}
    headers = {'Content-Type':'application/json;charset=utf8'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        response = json.loads(response.text)
        return response["data"]
    else:
        return -1


def slot_recognizer(text):
    """
    槽位识别器
    :param text:
    :return:
    """
    url = 'http://127.0.0.1:60061/service/api/medical_ner'
    data = {"text_list": [text]}
    headers = {'Content-Type': 'application/json;charset=utf8'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        response = json.loads(response.text)
        return response['data']
    else:
        return -1

def entity_link(mention, etype):
    """
    #TODO 对于识别到的实体mention,如果不是知识库中的标准称谓
    则对其进行实体链指，将其指向一个唯一实体
    :param mention:
    :param etype:
    :return:
    """
    return mention

def classifier(text):
    """
    判断是否是闲聊意图，以及是什么类型闲聊
    :param text:
    :return:
    """
    return clf_model.predict(text)

def neo4j_searcher(cql_list):
    """
    知识图谱查询
    :param cql_list:
    :return:
    """
    ress = ""
    if isinstance(cql_list, list):
        for cql in cql_list:
            rst = []
            data = graph.run(cql).data()
            if not data:
                continue
            for item in data:
                item_values = list(item.values())
                if isinstance(item_values[0], list):
                    rst.extend(item_values[0])
                else:
                    rst.extend(item_values)
            data = "、".join([str(i) for i in rst])
            ress += data + "\n"
    else:
        data = graph.run(cql_list).data()
        # 这里要分情况：1、查到了，且不为空；2、查到了，但是结果是None([{'p.desc': None}] )；3、直接连不上数据库
        # 三种情况都要有对应的兜底处理
        if not data:
            return ress
        rst = []
        for item in data:
            item_values = list(item.values())
            if isinstance(item_values[0], list):
                rst.extend(item_values[0])
            else:
                rst.extend(item_values)
        data = "、".join([str(i) for i in rst])
        ress += data

    return ress

def semantic_parser(text, user):
    """
    对用户输入文本进行解析,然后填槽，确定回复策略
    :param text:
    :param user:
    :return:
            填充slot_info中的["slot_values"]
            填充slot_info中的["intent_strategy"]
    """
    # 对医疗意图进行二次分类
    intent_receive = intent_classifier(text) # {'confidence': 0.8997645974159241, 'intent': '治疗方法'}
    print("intent_receive:",intent_receive)
    # 实体识别
    slot_receive = slot_recognizer(text)
    print("slot_receive:", slot_receive)

    if intent_receive == -1 or slot_receive == -1 or intent_receive.get("intent")=="其他":
        return semantic_slot.get("unrecognized")

    print("intent:", intent_receive.get("intent"))
    slot_info = semantic_slot.get(intent_receive.get("intent"))
    print("slot_info:", slot_info)
    # 填槽
    slots = slot_info.get("slot_list") # ["Disease"]
    slot_values = {}
    for slot in slots:              # 遍当前意图下的所有槽位,可以设置多个槽位解决任务型问答
        slot_values[slot] = None    # 将槽位置空
        for entities_info in slot_receive:
            for entity in entities_info["entities"]:
                if slot.lower() == entity["type"]:
                    slot_values[slot] = entity_link(entity["word"], entity["type"])

    last_slot_values = load_user_dialogue_context(user)["slot_values"]
    for k in slot_values.keys():
        if slot_values[k] is None:
            slot_values[k] = last_slot_values.get(k, None)
    slot_info["slot_values"] = slot_values

    # 根据意图的置信度来确认回复策略
    # TODO 使用强化学习进行策略选择
    conf = intent_receive.get("confidence")
    if conf >= intent_threshold_config["accept"]:   # >0.8
        slot_info["intent_strategy"] = "accept"
    elif conf >= intent_threshold_config["deny"]:   # >0.4
        slot_info["intent_strategy"] = "clarify"
    else:
        slot_info["intent_strategy"] = "deny"

    print("semantic_parser:",slot_info)
    return slot_info


def get_answer(slot_info):
    """
    根据不同的回复策略，去neo4j中查询答案
    :param slot_info:
    :return: 在slot_info中增加"replay_answer"这一项
    """
    cql_template = slot_info.get("cql_template")
    reply_template = slot_info.get("reply_template")
    ask_template = slot_info.get("ask_template")
    slot_values = slot_info.get("slot_values")
    strategy = slot_info.get("intent_strategy")

    if not slot_values:
        return slot_info

    if strategy == "accept":
        cql_list = []
        if isinstance(cql_template, list):
            for cql in cql_template:
                cql_list.append(cql.format(**slot_values)) # 通过字典设置参数
        else:
            cql_list = cql_template.format(**slot_values)

        answer = neo4j_searcher(cql_list)
        print("neo4j result for accept:", answer)
        if not answer:
            slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"

        elif  answer=="None":
            slot_info["replay_answer"] = "数据库中没有查到相关内容哦~"
        else:
            pattern = reply_template.format(**slot_values)
            slot_info["replay_answer"] = pattern + answer

    elif strategy == "clarify":
        # 0.4 < 意图置信度 < 0.8时，进行问题澄清
        pattern = ask_template.format(**slot_values)
        print("pattern for clarity:", pattern)

        slot_info["replay_answer"] = pattern
        # 得到肯定意图之后，需要给出用户回复的答案
        cql_list = []
        if isinstance(cql_template, list):
            for cql in cql_template:
                cql_list.append(cql.format(**slot_values))
        else:
            cql_list = cql_template.format(**slot_values)

        answer = neo4j_searcher(cql_list)
        print("neo4j result for clarify:", answer)

        if not answer:
            slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"

        elif  answer=="None":
            slot_info["choice_answer"] = "数据库中没有查到相关内容哦~"

        else:
            pattern = reply_template.format(**slot_values)
            slot_info["choice_answer"] = pattern + answer

    elif strategy == "deny":
        slot_info["replay_answer"] = slot_info.get("deny_response")

    print("get_answer:", slot_info)
    return slot_info

def chitchat_bot(intent):
    """
    如果是闲聊，就从闲聊的回复语料里随机选择一个返回给用户
    :param intent:
    :return:
    """
    return random.choice(chitchat_corpus.get(intent))

def medical_bot(text, user):
    """
    如果确定是诊断意图，则使用该函数进行诊断问答
    :param text:
    :param user:
    :return:
    """
    semantic_slot = semantic_parser(text, user)
    answer = get_answer(semantic_slot)
    return answer




