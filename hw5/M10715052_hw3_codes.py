import os
import math
import cupy as cp
import numpy as np
from numba import jit
import pandas as pd
import time

t1 = time.time()
topic_k = 512
all_word_index = {}
bglm = {}
col = np.zeros([51253, 18461], dtype=int)
col_wc = np.zeros(18461, dtype=int)
WiTk = cp.random.rand(51253, topic_k)
WiTk = WiTk / cp.sum(WiTk, axis=0)

TkDj = cp.random.rand(topic_k, 20726)
TkDj = TkDj / cp.sum(TkDj, axis=0)

t2 = time.time()
print("1st phase finished with %.2f sec." % (t2 - t1))


def word_counting(doc, serial, index_array):
    for i in range(len(doc)):
        index_array[int(doc[i]), serial] += 1


# =========================================前處理========================================
fc = open('BGLM.txt')
lines = fc.readlines()
for line in lines:
    temp = np.array(line.split())
    bglm[int(temp[0])] = math.exp(float(temp[1]))
fc.close()

fc = open('Collection.txt')
lines = fc.readlines()
for i in range(len(lines)):
    temp = np.array(lines[i].split())
    word_counting(temp, i, col)
    col_wc[i] = len(temp)
fc.close()

temp = None
lines = None

docs = np.array(os.listdir("Document"))
qrys = np.array(os.listdir("Query"))
docs_index = np.zeros([51253, 2265], dtype=int)
docs_wc = np.zeros([2265], dtype=int)
for i in range(len(docs)):
    fd = open('Document\\'+docs[i])
    lines = fd.readlines()[3:]
    temp = []
    for line in lines:
        temp += line.split()[:-1]
    word_counting(temp, i, docs_index)
    docs_wc[i] += len(temp)
    fd.close()

qries_index = [] #裡面裝每個query的dict
for i in range(len(qrys)):
    fq = open('Query\\' + qrys[i])
    lines = fq.readlines()
    temp_dict = {}
    temp = []
    for line in lines:
        temp += line.split()[:-1]
    for item in temp:
        if item not in temp_dict:
            temp_dict[int(item)] = 1
        else:
            temp_dict[int(item)] += 1
    qries_index.append(temp_dict)

col = np.append(col, docs_index, 1)
col_wc = np.append(col_wc, docs_wc)
docs_index = None
docs_wc = None
temp = None
lines = None
temp_dict = None
t3 = time.time()
print("2nd phase finished with %.2f sec." % (t3 - t2))


# =========================================EM_COL========================================

col = cp.asarray(col)
col_wc = cp.asarray(col_wc)
for EM_round in range(67):  # train幾次
    t3 = time.time()
    new_WiTk = cp.zeros([51253, topic_k], dtype=cp.float)
    new_TkDj = cp.zeros([20726, topic_k], dtype=cp.float)
    for j in range(20726):
        temp_WiTk = WiTk * TkDj[:, j]
        x = cp.sum(temp_WiTk, axis=1).reshape(51253, 1)
        x[x == 0] = 1
        temp_WiTk /= x
        # temp_WiTk = cp.asarray(np.nan_to_num(cp.asnumpy(temp_WiTk)))
        temp_WiTk *= col[:, j].reshape(51253, 1)
        new_WiTk += temp_WiTk
        new_TkDj[j] = cp.sum(temp_WiTk, axis=0) / col_wc[j]
    WiTk = new_WiTk / cp.sum(new_WiTk, axis=0)
    WiTk = cp.asarray(np.nan_to_num(cp.asnumpy(WiTk)))
    TkDj = new_TkDj.swapaxes(0, 1)
    TkDj = cp.asarray(np.nan_to_num(cp.asnumpy(TkDj)))
    t4 = time.time()
    print("3rd phase %d finished with %.2f sec." % (EM_round, (t4 - t3)))
new_WiTk = None
new_TkDj = None
temp_WiTk = None

# -----------------------------------FOLD_IN-------------------------------------------------------
'''
TkDj = cp.random.rand(topic_k, 20726)
TkDj = TkDj / cp.sum(TkDj, axis=0)
t3 = time.time()
print("4th phase %d finished with %.2f sec." % (EM_round, (t3 - t4)))

for EM_round in range(50):
    t4 = time.time()
    new_TD = cp.zeros([2265, topic_k], dtype=cp.float)
    for j in range(2265):
        TD = WiTk * TkDj[:, j]
        x = cp.sum(TD, axis=1).reshape(51253, 1)
        x[x == 0] = 1
        TD /= x
        TD *= docs_index[:, j].reshape(51253, 1)
        new_TD[j] = cp.sum(TD, axis=0) / docs_wc[j]

    TkDj = new_TD.swapaxes(0, 1)
    t5 = time.time()
    print("5th phase %d finished with %.2f sec." % (EM_round, (t5 - t4)))
new_TD = None
TD = None
# 算完之後，WD矩陣即為所求之值
# ====================================寫檔==============================================
'''

fw = open("result3.2.txt", "w")
fw.write("Query,RetrievedDocuments")
for qry_num in range(len(qrys)):
    docs_p = np.zeros(2265)
    for j in range(2265):  # 18461~20716 total 2256
        sum_of_log = 0
        for item in qries_index[qry_num]:
            temp = (WiTk[item] * TkDj[:, j+18461])

            #temp /= cp.sum(temp)

            sum_of_p = 0.28 * (col[item, j+18461] / col_wc[j+18461]) + 0.62 * cp.sum(temp) + 0.1 * bglm[item]
            sum_of_log += math.log(sum_of_p)
        docs_p[j] = sum_of_log

    score_dict = {'score': docs_p}
    score_df = pd.DataFrame(score_dict)
    score_df = score_df.sort_values(by=['score'], ascending=False)
    SS = score_df.index
    fw.write('\n' + qrys[qry_num] + ',')
    for lss in SS:
        fw.write(docs[lss] + ' ')

t6 = time.time()
#print("6th phase finished with %.2f sec." % (t6 - t5))
print("Total cost %.2f sec." % (t6 - t1))
fw.close()
