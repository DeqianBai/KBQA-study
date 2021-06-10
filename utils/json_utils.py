#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2013-2020, Node Supply Chain Manager Corporation Limited.
@time: 2021/6/4 11:06
@file: json_utils.py
@author: baidq
@Software: PyCharm
@desc:
'''

import os
import re
import json

LOGS_DIR = "./logs"

def dump_user_dialogue_context(user, data):
    """

    :param user:
    :param data:
    :return:
    """
    path = os.path.join(LOGS_DIR, "{}.json".format(str(user)))
    with open(path, "w", encoding="utf8") as f:
        f.write(json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))

def load_user_dialogue_context(user):
    """

    :param user:
    :return:
    """
    path = os.path.join(LOGS_DIR, '{}.json'.format(str(user)))
    if not os.path.exists(path):
        return  {"choice_answer":"hi，机器人小智很高心为您服务","slot_values":None}
    else:
        with open(path, 'r', encoding='utf8') as f:
            data = f.read()
            return json.loads(data)