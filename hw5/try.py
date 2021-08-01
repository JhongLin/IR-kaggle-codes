import os
import math
import cupy as cp
import numpy as np
import pandas as pd
import time



def word_counting(doc, serial, index_array):
    for i in range(len(doc)):
        index_array[int(doc[i]), serial] += 1

def remove_zero_rows(X):
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]
def remove_zero_rows_cupy(X):
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = cp.nonzero(X)
    unique_nonzero_indice = cp.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]

docs = np.array(os.listdir("Document"))
qrys = np.array(os.listdir("Query"))

fv = open('vec.txt')
lines = fv.readlines()

vec = np.zeros([51253, 100], dtype=float)
score = np.zeros([len(qrys), len(docs)])


temp = []
for line in lines:
    temp = line.split()
    serial = int(temp[0])
    temp = temp[1:]
    temp = np.array(temp).astype(np.float)
    vec[serial] += temp

fv.close()


docs_index = np.zeros([51253, len(docs)], dtype=float)
unid_index = np.zeros([51253, len(docs)], dtype=int)
docs_wc = np.zeros(len(docs), dtype=int)
uni_docs = []  #裡面裝每個doc的uniWord
for i in range(len(docs)):
    fd = open('Document\\' + docs[i])
    lines = fd.readlines()[3:]
    temp = []
    temp_dict = {}
    for line in lines:
        temp += line.split()[:-1]
    word_counting(temp, i, docs_index)
    docs_wc[i] += len(temp)
    for item in temp:
        if int(item) not in temp_dict:
            temp_dict[int(item)] = 1
        else:
            temp_dict[int(item)] += 1
    temp = []
    for item in temp_dict:
        temp.append(int(item))
        unid_index[int(item), i] = 1
    temp = np.sort(np.array(temp))
    uni_docs.append(temp)
    fd.close()

Pwid = docs_index/docs_wc
pwd_list = []
for i in range(len(docs)):
    temp = np.array([])
    for j in range(len(uni_docs[i])):
        temp = np.append(temp, Pwid[uni_docs[i][j], i])
    pwd_list.append(temp)

uniq_index = np.zeros([51253, len(qrys)], dtype=int)
qrys_index = [] #裡面裝每個query的dict
uni_qrys = [] #裡面裝每個query的uniWord
for i in range(len(qrys)):
    fq = open('Query\\' + qrys[i])
    lines = fq.readlines()
    temp_dict = {}
    temp = []
    for line in lines:
        temp += line.split()[:-1]
    for item in temp:
        if int(item) not in temp_dict:
            temp_dict[int(item)] = 1
        else:
            temp_dict[int(item)] += 1
    qrys_index.append(temp_dict)
    temp=[]
    for item in temp_dict:
        temp.append(item)
        uniq_index[int(item), i] = 1
    temp = np.array(temp)
    temp = np.sort(temp)
    uni_qrys.append(temp)
    fq.close()


for j in range(len(qrys)):
    t1 = time.time()
    vec_q = uniq_index[:,j].reshape(51253, 1)*vec
    vec_q = remove_zero_rows(vec_q)

    for i in range(len(docs)):
        vec_d = unid_index[:,i].reshape(51253, 1)*vec
        vec_d = remove_zero_rows(vec_d)
        vec_d = vec_d.swapaxes(0, 1)
        Pji = np.exp(np.dot(vec_q, vec_d))
        Pji /= np.sum(Pji, axis=0)
        Pji *= pwd_list[i]
        Pji = np.sum(Pji, axis=1)
        score[j, i] = np.multiply.reduce(Pji)
    print(j)
    t2 = time.time()
    print("with %.2f sec." % (t2 - t1))
