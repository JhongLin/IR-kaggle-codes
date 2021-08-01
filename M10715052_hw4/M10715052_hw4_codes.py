import os
import math
import cupy as cp
import numpy as np
# from numba import jit
import pandas as pd
import time

# ===========================================PHASE 1======================================================
N = 2265
t1 = time.time()

def word_counting(doc, serial, index_array):
    for i in range(len(doc)):
        index_array[int(doc[i]), serial] += 1

docs = np.array(os.listdir("Document"))
qrys = np.array(os.listdir("Query"))
docs_index = np.zeros([51253, 2265], dtype=float)
# docs_wc = np.zeros([2265], dtype=int)
for i in range(len(docs)):
    fd = open('Document\\' + docs[i])
    lines = fd.readlines()[3:]
    temp = []
    for line in lines:
        temp += line.split()[:-1]
    word_counting(temp, i, docs_index)
    # docs_wc[i] += len(temp)
    fd.close()
qrys_tw = np.zeros([51253, len(qrys)])
qrys_index = []  # 裡面裝每個query的dict
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
    fq.close()
    word_counting(temp, i, qrys_tw)

temp = None
temp_dict = None
lines = None

ni = docs_index.copy()
ni[ni > 1] = 1
ni = np.sum(ni, axis=1)
ni[ni == 0] = N
idf = np.log(ni) * (-1) + math.log(N)
max_iq = np.zeros(len(qrys), dtype=int)
for i in range(len(max_iq)):
    max_wordName, max_amount = max(qrys_index[i].items(), key=lambda x: x[1])
    max_iq[i] = max_amount

t2 = time.time()
print("1st phase finished with %.2f sec." % (t2 - t1))
# ===========================================PHASE 2======================================================

docs_index += 0.8
docs_index[docs_index==0.8]=0
qrys_tw =(qrys_tw) + 1.5
qrys_tw[qrys_tw==1.5]=0
R_num=25
qry_topK_doc = cp.zeros([len(qrys), R_num])
docs_tw = cp.asarray( (docs_index)* idf.reshape(51253, 1))# * idf.reshape(51253, 1)
docs_index = None
qrys_tw = cp.asarray( (qrys_tw)* idf.reshape(51253, 1))# * idf.reshape(51253, 1)

for i in range(len(qrys)):
    cosine = cp.sum(docs_tw*qrys_tw[:, i].reshape(51253, 1), axis=0)/ (cp.linalg.norm(docs_tw,axis=0) * cp.linalg.norm(qrys_tw[:, i]))
    qry_topK_doc[i] = cp.argsort(cosine*(-1))[0:R_num]

t3 = time.time()
print("2nd phase finished with %.2f sec." % (t3 - t2))
# ===========================================PHASE 3======================================================
training_times = 3
for round in range(training_times):
    t3 = time.time()
    new_qrys_tw = cp.zeros([len(qrys), 51253])
    RLV_doc = cp.zeros([R_num, 51253])
    for i in range(len(qrys)):
        for j in range(R_num):
           RLV_doc[j] = docs_tw[:, int(qry_topK_doc[i, j])]
        new_qrys_tw[i] = 0.15*qrys_tw[:, i] + 0.85*(cp.sum(RLV_doc, axis=0)/R_num)
    new_qrys_tw = new_qrys_tw.swapaxes(0, 1)
    if round == training_times-1:
        break
    for i in range(len(qrys)):
        cosine = cp.sum(docs_tw * new_qrys_tw[:, i].reshape(51253, 1), axis=0) / (cp.linalg.norm(docs_tw, axis=0) * cp.linalg.norm(new_qrys_tw[:, i]))
        qry_topK_doc[i] = cp.argsort(cosine*(-1))[0:R_num]
    t4 = time.time()
    print("3rd phase round %d finished with %.2f sec." % (round, (t4 - t3)))
t4 = time.time()
# ===========================================PHASE 4======================================================

fw = open("hw4cos_ex.txt", "w")
for i in range(len(qrys)):
    cosine = cp.sum(docs_tw*new_qrys_tw[:, i].reshape(51253, 1), axis=0)/ (cp.linalg.norm(docs_tw,axis=0) * cp.linalg.norm(new_qrys_tw[:, i]))
    temp_rank = cp.argsort(cosine * (-1))[0:100]
    for j in range(len(cosine)):
        fw.write(str(cosine[j])+' ')
    fw.write('\n')

fw.close()
t5 = time.time()
print("4td phase finished with %.2f sec." % (t5 - t4))
