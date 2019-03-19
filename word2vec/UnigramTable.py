#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Date : 2019/3/8 20:22 
# @Author : maoss 
# @File : UnigramTable.py
import math
import numpy as np

class UnigramTable():
    """
    生成负采样概率表
    """
    def __init__(self, word_dict):
        vocab_size = word_dict.__len__()
        power = 0.75
        norm = sum([math.pow(word['freq'], power) for word in word_dict.values()])
        table_size = 1000 # 1e8 ??
        table = np.zeros(table_size, dtype=np.uint32)
        p = 0
        i = 0
        for j, unigram in enumerate(word_dict.values()):
            p += float(math.pow(unigram['freq'], power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=self.table.__len__(), size=count)
        return [self.table[i] for i in indices]


