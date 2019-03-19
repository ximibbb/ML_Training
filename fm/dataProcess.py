
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[29]:


# 读取featureIndex数据，统计基本的信息，field等
FIELD_SIZES = [0] * 26
with open('../data/featindex.txt') as fin:
    for line in fin:
        line = line.strip().split(':')
        if len(line) > 1:
            featIndex = int(line[0]) - 1
            FIELD_SIZES[featIndex] += 1
   
FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = sum(FIELD_SIZES)
OUTPUT_DIM = 1

print('field sizes:', FIELD_SIZES)
print('INPUT_DIM:', INPUT_DIM)


# In[30]:


# 读取libsvm格式数据成稀疏矩阵形式
# 0 5:1 9:1 140858:1 445908:1 446177:1 446293:1 449140:1 490778:1 491626:1 491634:1 491641:1 491645:1 491648:1 491668:1 491700:1 491708:1
def read_data(file_name):
    X = []
    D = []
    y = []
    with open(file_name) as fin:
        for line in fin:
            fields = line.strip().split()
            y_i = int(fields[0])
            X_i = [int(x.split(':')[0]) for x in fields[1:]]
            D_i = [int(x.split(':')[1]) for x in fields[1:]]
            y.append(y_i)
            X.append(X_i)
            D.append(D_i)
    y = np.reshape(np.array(y), [-1])
    X = libsvm_2_coo(zip(X, D), (len(X), INPUT_DIM)).tocsr()
    return X, y


# In[31]:


# 工具函数，libsvm格式转成coo稀疏存储格式
def libsvm_2_coo(libsvm_data, shape):
    coo_rows = []
    coo_cols = []
    coo_data = []
    n = 0
    for x, d in libsvm_data:
        coo_rows.extend([n] * len(x))
        coo_cols.extend(x)
        coo_data.extend(d)
        n += 1
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.array(coo_data)
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)

