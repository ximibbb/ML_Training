#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Date : 2019/3/8 16:44 
# @Author : maoss 
# @File : word2vec.py

import jieba
from WordCount import WordCounter
from HuffmanTree import HuffmanTree
from UnigramTable import UnigramTable
import numpy as np
import math
from sklearn import preprocessing
from multiprocessing import Array

class Word2Vec():
    def __init__(self, vec_len=15000, learning_rate=0.01, win=5, model='cbow', method='hs', n_sampling=5):
        self.vec_len = vec_len
        self.learning_rate = learning_rate
        self.win = win
        self.model = model
        self.cutted_text_list = None
        self.word_dict = None
        self.huffman = None
        self.method = method
        self.n_sampling = n_sampling

    def train_model(self, text_list):
        before = (self.win - 1) >> 1
        after = self.win - 1 - before
        if self.method == 'hs':
            if self.huffman == None:
                if self.word_dict == None:
                    wc = WordCounter(text_list)
                    self.gen_word_dict(wc.count_res.larger_than(2)) # get self.word_dict
                    self.cutted_text_list = wc.text
                self.huffman = HuffmanTree(self.word_dict, vec_len=self.vec_len)
            print ("get word_dict and huffman tree, ready to train vector!")

            # start to train
            if self.model == 'cbow':
                print("==========CBOW===========")
                method = self.deal_cbow
            elif self.model == 'skip-gram':
                print("==========Skip-Gram===========")
                method = self.deal_skipGram
            if self.cutted_text_list:
                count = 0
                for line in self.cutted_text_list:
                    line_len = line.__len__()
                    for i in range(line_len):
                        method(line[i], line[max(0, i-before):i] + line[i+1:min(line_len, i+after+i)])
                    count += 1
                    print('{c}/{d}'.format(c=count, d=self.cutted_text_list.__len__()))
            else:
                print ("ERROR: cutted_text_list has not be generate!")
        else:
            if self.word_dict == None:
                wc = WordCounter(text_list)
                self.gen_word_dict(wc.count_res.larger_than(2))  # get self.word_dict
                self.cutted_text_list = wc.text

            table = UnigramTable(self.word_dict)
            if self.cutted_text_list:
                for line in self.cutted_text_list:
                    syn1 = np.zeros([1, self.vec_len])  # vocab_size = self.word_dict.__len__()
                    line_len = line.__len__()
                    for i in range(line_len):
                        gram_word_list = line[max(0, i-before):i] + line[i+1:min(line_len, i+after+i)]
                        for i in range(gram_word_list.__len__())[::-1]:
                            if not self.word_dict.__contains__(gram_word_list[i]):
                                gram_word_list.pop(i)
                        if gram_word_list.__len__() == 0:
                            return
                        # print (gram_word_list)
                        neu1 = np.mean(np.array([self.word_dict[word]['vector'] for word in gram_word_list]), axis=0)  # syn0: self.word_dict[word]['vector']
                        neu1e = np.zeros([1, self.vec_len])  # init e
                        classifiers = [(line[i], 1)] + [(line[target], 0) for target in table.sample(self.n_sampling)] # 负采样有问题？
                        for target, label in classifiers:
                            # print (target)
                            # print (neu1.shape, syn1.shape)
                            q = self.sigmoid(np.dot(neu1, syn1.T))
                            g = self.learning_rate * (label - q)
                            neu1e += g * syn1
                            syn1 += g * neu1e
                        # update syn0
                        for gram_word in gram_word_list:
                            self.word_dict[gram_word]['vector'] += neu1e
        print("训练的词向量为：")
        for word, value in self.word_dict.items():
                print (word, value['vector'])


    def gen_word_dict(self, word_count):
        word_dict = dict()
        if isinstance(word_count, dict):
            sum_count = sum(word_count.values())
            for word in word_count:
                print("gen_word_dict: ", word)
                temp_dict = dict(
                    word = word,
                    freq = word_count[word],
                    weight = word_count[word]/sum_count,
                    vector = np.random.random([1, self.vec_len]),
                    huffman = None
                )
                word_dict[word] = temp_dict
        else:
            word_count_list = [x[1] for x in word_count]
            sum_count = sum(word_count_list)
            for item in word_count:
                temp_dict = dict(
                    word = item[0],
                    freq = item[1],
                    weight = item[1]/sum_count,
                    vector = np.random.random([1, self.vec_len]),
                    huffman = None
                )
                word_dict[item[0]] = temp_dict
        self.word_dict = word_dict

    def deal_cbow(self, word, gram_word_list):
        # 获得 neu1
        if not self.word_dict.__contains__(word):
            # print ("word_dict not contains: ", word)
            return
        word_huffman = self.word_dict[word]['huffman']
        # print("word: ", word, "\tword_huffman: ", word_huffman)
        gram_vector_sum = np.zeros([1, self.vec_len])
        for i in range(gram_word_list.__len__())[::-1]:
            item = gram_word_list[i]
            if self.word_dict.__contains__(item):
                gram_vector_sum += self.word_dict[item]['vector']
            else:
                gram_word_list.pop(i)
        if gram_word_list.__len__() == 0:
            return
        e = self.update(word_huffman, gram_vector_sum, self.huffman.root)
        # 更新词向量
        for item in gram_word_list:
            self.word_dict[item]['vector'] += e
            self.word_dict[item]['vector'] = preprocessing.normalize(self.word_dict[item]['vector'])

    def deal_skipGram(self, word, gram_word_list):
        if not self.word_dict.__contains__(word):
            # print ("word_dict not contains: ", word)
            return
        # 由上下文预测中心词
        word_vec = self.word_dict[word]['vector']
        for i in range(gram_word_list.__len__())[::-1]:
            if not self.word_dict.__contains__(gram_word_list[i]):
                gram_word_list.pop(i)
        if gram_word_list.__len__() == 0:
            return
        for u in gram_word_list:
            u_huffman = self.word_dict[u]['huffman']
            e = self.update(u_huffman, word_vec, self.huffman.root)
            self.word_dict[word]['vector'] += e
            self.word_dict[word]['vector'] = preprocessing.normalize(self.word_dict[word]['vector'])


    def update(self, word_huffman, input_vec, root):
        node = root
        # print ("node.value: ", node.value)
        e = np.zeros([1, self.vec_len])
        for level in range(word_huffman.__len__()):
            huffman_charat = word_huffman[level] # huffman_charat -> d_j^w
            q = self.sigmoid(input_vec.dot(node.value.T)) # input_vec -> X_w(neu1), node.value -> theta_j(syn1)
            grad = self.learning_rate * (1 - int(huffman_charat) - q) # 计算梯度
            e += grad * node.value # neu1e
            node.value += grad * input_vec # 更新非叶子节点
            # print ("update node.value: ", node.value)
            node.value = preprocessing.normalize(node.value) # 归一化
            if huffman_charat == '0':
                node = node.right
            else:
                node = node.left
        return e

    def sigmoid(self, value):
        return 1/(1+math.exp(-value)) # 还有近似算法

    def expTabel(self):
        pass


if __name__ == '__main__':
    data = ['Merge multiple sorted inputs into a single sorted output',
            'The API below differs from textbook heap algorithms in two aspects',
            'Merge multiple sorted inputs']
    # wv = Word2Vec(vec_len=100, model='skip-gram', method='hs')
    wv = Word2Vec(vec_len=100, model='skip-gram', method='neg')
    wv.train_model(data)