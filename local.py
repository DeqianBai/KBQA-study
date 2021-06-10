#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/6/7 9:48
@file: local.py
@author: baidq
@Software: PyCharm
@desc:
'''

import os
import itchat
import json
from sanic import Sanic, response
from sanic_cors import CORS
from sanic_openapi import swagger_blueprint, doc

from modules import chitchat_bot,medical_bot, classifier
from utils.json_utils import dump_user_dialogue_context, load_user_dialogue_context

"""
问答流程：
1、用户输入文本
2、对文本进行解析得到语义结构信息
3、根据语义结构去查找知识，返回给用户

文本解析流程：
1、意图识别
    闲聊意图：greet, goodbye, accept, deny, isbot
            greet, goodbye: 需要有回复动作
            accept, deny：需要执行动作
    诊断意图：
            当意图置信度达到一定阈值时(>=0.8)，可以查询该意图下的答案
            当意图置信度较低时(0.4~0.8)，按最高置信度的意图查找答案，反问用户，进行问题澄清
            当意图置信度更低时(<0.4)，拒绝回答
2、槽位填充
    如果输入是一个诊断意图，那么就需要填充语义槽，得到结构化语义
"""

def delete_cache(file_name):
    """
    清除缓存数据，切换账号登入
    :param file_name:
    :return:
    """
    if os.path.exists(file_name):
        os.remove(file_name)

@itchat.msg_register(["Text"])
def text_replay(msg):
    """
    微信入口
    :param msg:
    :return:
    """
    user_intent= classifier(msg["Text"])
    print("user_intent:",user_intent)
    if user_intent in ["greet","goodbye","deny","isbot"]:
        reply = chitchat_bot(user_intent)
    elif user_intent == "accept":
        reply = load_user_dialogue_context(msg.User['NickName'])
        reply = reply.get("choice_answer")
    else:
        reply = medical_bot(msg["Text"], msg.User['NickName'])
        if reply["slot_values"]:
            dump_user_dialogue_context(msg.User['NickName'],reply)
        reply = reply.get("replay_answer")
    msg.user.send(reply)



app = Sanic(__name__)
CORS(app)

app.blueprint(swagger_blueprint)
app.config["API_VERSION"] = "0.1"
app.config["API_TITLE"] = "DIALOG_SERVICE：Sanci-OpenAPI"

server_port = int(os.getenv('SERVER_PORT', 12348))
@app.post("/bot/message")
@doc.summary("Let us have a chat")
@doc.consumes(doc.JsonBody({"message": str}), location="body")
def message(request):
    """
    sanic 入口
    :param request:
    :return:
    """
    # 获取用户ID和用户输入
    sender = request.json.get("sender")
    message = request.json.get("message")
    print("sender:{}, message:{}".format(sender, message))
    # 判断用户意图是否属于闲聊类，相当于第一层意图过滤
    user_intent = classifier(message)
    print("user_intent:", user_intent)
    if user_intent in ["greet", "goodbye", "deny", "isbot"]:
        reply = chitchat_bot(user_intent)
    elif user_intent == "accept":
        reply = load_user_dialogue_context(sender)
        reply = reply.get("choice_answer")
        print("01-accept:",reply)
    # diagnosis
    else:
        reply = medical_bot(message, sender)
        if reply["slot_values"]:
            dump_user_dialogue_context(sender,reply)
        reply = reply.get("replay_answer")
    print("reply:",reply)
    return response.json(reply)

if __name__ == '__main__':
    """
    测试用例：
    你好
    你是机器人吗
    请问得了心脏病怎么办呢
    心肌炎是什么
    怎么治疗较好呢
    需要治疗多久才会恢复呢？？
    是的
    
    """
    # 打开下面注释可以清除对话日志缓存
    # delete_cache(file_name='./logs/loginInfo.pkl')

    # 打开下面日志使用微信进行对话
    # itchat.auto_login(hotReload=True, enableCmdQR=2, statusStorageDir='./logs/loginInfo.pkl')
    # itchat.run()

    # 打开下面对话使用swagger在网页端进行对话
    app.run(host="192.168.0.224",port=server_port)