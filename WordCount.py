#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Date : 2019/3/8 17:01 
# @Author : maoss 
# @File : WordCount.py

import jieba
from collections import Counter
from operator import itemgetter as _itemgetter


class WordCounter():
    def __init__(self, text):
        self.text = text
        self.word_count(self.text)

    def word_count(self, text, cut_all=False):
        count = 0
        word_list = list()
        for line in text:
            res = jieba.cut(line, cut_all=cut_all)
            res = list(res)
            text[count] = res
            count += 1
            word_list += res
        # print (word_list)
        self.count_res = filter_counter(word_list)
        print (self.count_res)


class filter_counter(Counter):
    def __init__(self, element_list):
        super().__init__(element_list)

    def larger_than(self, minValue, ret='list'):
        """
        低值过滤
        """
        temp = sorted(self.items(), key=_itemgetter(1), reverse=True)
        low = 0
        high = temp.__len__()
        while(high - low > 1):
            mid = (low + high) >> 1
            if temp[mid][1] >= minValue:
                low = mid
            else:
                high = mid
        if temp[low][1] < minValue:
            if ret == 'dict':
                return dict()
            else:
                return list()
        if ret == 'dict':
            ret_data = dict()
            for elem, count in temp[:high]:
                ret_data[elem] = count
            return ret_data
        else:
            return temp[:high]

    def less_than(self, maxValue, ret='list'):
        """
        高值过滤
        """
        temp = sorted(self.items(), key=_itemgetter(1)) # 从小到大
        low = 0
        high = temp.__len__()
        while(high - low > 1):
            mid = high + low >> 1
            if temp[mid][1] < maxValue:
                low = mid
            else:
                high = mid
        if temp[low][1] > maxValue:
            if ret == 'dict':
                return dict()
            else:
                return list()
        if ret == 'dict':
            ret_data = dict()
            for elem, count in temp[:high]:
                ret_data[elem] = count
            return ret_data
        else:
            return temp[:high]

if __name__ == '__main__':
    data = ['Merge multiple sorted inputs into a single sorted output',
            'The API below differs from textbook heap algorithms in two aspects']
    wc = WordCounter(data)
    c = wc.count_res
    print (c)