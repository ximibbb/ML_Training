#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Date : 2019/3/8 16:00 
# @Author : maoss
# @File : HuffmanTree.py

import numpy as np

class HuffmanTreeNode():
    def __init__(self, value, weight):
        self.right = None
        self.left = None
        self.value = value
        self.weight = weight
        self.huffmanCode = ""

class HuffmanTree():
    def __init__(self, word_dict, vec_len=15000):
        self.vec_len = vec_len
        self.root = None
        word_dict_list = list(word_dict.values())
        node_list = [HuffmanTreeNode(x['word'], x['weight']) for x in word_dict_list]
        self.build_tree(node_list) # 生成Huffman Tree
        self.gen_huffman_code(self.root, word_dict) # 生成Huffman编码

    def gen_huffman_code(self, root, word_dict):
        tree_stack = [self.root]
        while (tree_stack.__len__() > 0):
            node = tree_stack.pop()
            while node.left or node.right:
                code = node.huffmanCode
                node.left.huffmanCode = code + "1"
                node.right.huffmanCode = code + "0"
                tree_stack.append(node.right)
                node = node.left
            word = node.value
            code = node.huffmanCode
            word_dict[word]['huffman'] = code
            print(word, '\t', code, '\t', node.weight)


    def build_tree(self, node_list):
        while node_list.__len__() > 1: # len() / __len()__
            node1 = 0
            node2 = 1 # node1, node2表示概率最小的两个节点
            if node_list[node2].weight < node_list[node1].weight:
                [node1, node2] = [node2, node1]
            for i in range(2, node_list.__len__()):
                if node_list[i].weight < node_list[node2].weight:
                    node2 = i
                    if node_list[node2].weight < node_list[node1].weight:
                        [node1, node2] = [node2, node1]
            new_node = self.merge(node_list[node1], node_list[node2])
            if node1 < node2:
                node_list.pop(node2)
                node_list.pop(node1)
            elif node1 > node2:
                node_list.pop(node1)
                node_list.pop(node2)
            else:
                raise RuntimeError("node1, node2 has the same position")
            node_list.insert(0, new_node)
        self.root = node_list[0]

    def merge(self, node1, node2):
        new_weight = node1.weight + node2.weight
        new_node = HuffmanTreeNode(np.zeros([1, self.vec_len]), new_weight)
        if node1.weight < node2.weight:
            new_node.left = node1
            new_node.right = node2
        else:
            new_node.left = node2
            new_node.right = node1
        return new_node