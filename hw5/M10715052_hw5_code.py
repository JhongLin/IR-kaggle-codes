import os
import math
import cupy as cp
import numpy as np
import pandas as pd
import time
from sklearn.externals import joblib


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
vec_size = 256
vec_n,vec_v=joblib.load('word2vec256_w13_e100_n5.vec')
vec_n = np.array(vec_n).astype(int)
vec_v = np.array(vec_v).astype(float)
dv = np.zeros([51253, vec_size], dtype=float)
for i in range(len(vec_n)):
    dv[vec_n[i]] += vec_v[i]

t0 = time.time()
docs = np.array(os.listdir("Document"))
qrys = np.array(os.listdir("Query"))

fv = open('vec.txt')
lines = fv.readlines()

vec = np.zeros([51253, 100], dtype=float)
score = np.zeros([len(qrys), len(docs)])
score_sec = np.zeros([len(qrys), len(docs)])

temp = []
for line in lines:
    temp = line.split()
    serial = int(temp[0])
    temp = temp[1:]
    temp = np.array(temp).astype(np.float)
    vec[serial] += temp
fv.close()


bglm = {}
fb = open('BGLM.txt')
lines = fb.readlines()
for line in lines:
    temp = np.array(line.split())
    bglm[int(temp[0])] = math.exp(float(temp[1]))
fb.close()



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
    temp = cp.asarray(temp)
    pwd_list.append(temp)

uniq_index = np.zeros([51253, len(qrys)], dtype=int)
qrys_index = np.zeros([51253, len(qrys)], dtype=float)
uni_qrys = [] #裡面裝每個query的uniWord
for i in range(len(qrys)):
    fq = open('Query\\' + qrys[i])
    lines = fq.readlines()
    temp_dict = {}
    temp = []
    for line in lines:
        temp += line.split()[:-1]
    word_counting(temp, i, qrys_index)
    for item in temp:
        if int(item) not in temp_dict:
            temp_dict[int(item)] = 1
        else:
            temp_dict[int(item)] += 1

    temp=[]
    for item in temp_dict:
        temp.append(item)
        uniq_index[int(item), i] = 1
    temp = np.array(temp)
    temp = np.sort(temp)
    uni_qrys.append(temp)
    fq.close()

#vec = cp.asarray(vec)
uniq_index = cp.asarray(uniq_index)
unid_index = cp.asarray(unid_index)
vec = cp.asarray(dv)

t1 = time.time()
print("1st phase finished with %.2f sec." % (t1 - t0))
#-------------------------------------------------------
t0 = time.time()
docs_rate_list = []
qrys_rate_list = []
docs_index = cp.asarray(docs_index)
qrys_index = cp.asarray(qrys_index)
docs_index /= cp.sum(docs_index, axis=0)
qrys_index /= cp.sum(qrys_index, axis=0)
for i in range(len(docs)):
    temp = docs_index[:,i].reshape(51253,1)
    temp = remove_zero_rows_cupy(temp)
    docs_rate_list.append(temp)
for i in range(len(qrys)):
    temp = qrys_index[:,i].reshape(51253,1)
    temp = remove_zero_rows_cupy(temp)
    qrys_rate_list.append(temp)
t1 = time.time()
print("2nd phase finished with %.2f sec." % (t1 - t0))
#---------------------------------------------------------

t1 = time.time()

qrys_vec = cp.zeros([len(qrys), vec_size])
docs_vec = cp.zeros([len(docs), vec_size])

for j in range(len(qrys)):
    vec_q = uniq_index[:,j].reshape(51253, 1)*vec
    vec_q = remove_zero_rows_cupy(vec_q)
    vec_q *= qrys_rate_list[j]
    vec_q = cp.sum(vec_q, axis=0)
    qrys_vec[j] += vec_q

for i in range(len(docs)):
    vec_d = unid_index[:,i].reshape(51253, 1)*vec
    vec_d = remove_zero_rows_cupy(vec_d)
    vec_d *= docs_rate_list[i]
    vec_d = cp.sum(vec_d, axis=0)
    docs_vec[i] += vec_d

a = cp.linalg.norm(qrys_vec, axis=1)
b = cp.linalg.norm(docs_vec, axis=1)
c = cp.dot(a.reshape(len(qrys), 1), b.reshape(1, len(docs)))

docs_vec = docs_vec.swapaxes(0, 1)
cosine = cp.dot(qrys_vec, docs_vec)/c

t2 = time.time()
print("3rd phase finished with %.2f sec." % (t2 - t1))

#------------------------------------------------------

t0 = time.time()
hw4_cos = np.zeros([len(qrys), len(docs)])

fh = open('hw4cos.txt')
lines = fh.readlines()

temp = []
for i in range(len(lines)):
    temp = lines[i].split()
    temp = np.array(temp).astype(np.float)
    hw4_cos[i] += temp
fh.close()


t1 = time.time()
print("4th phase finished with %.2f sec." % (t1 - t0))



fw = open("result2.03.txt", "w")
fw.write("Query,RetrievedDocuments")
for i in range(len(qrys)):
    rank = cp.argsort(cosine[i]*(-1))[0:100]
    fw.write('\n' + qrys[i] + ',')
    for s in rank:
        fw.write(docs[int(s)] + ' ')
fw.close()

cosine = cp.asnumpy(cosine)
nc = cosine * hw4_cos
fw = open("fot.txt", "w")
fw.write("Query,RetrievedDocuments")
for i in range(len(qrys)):
    rank = np.argsort(nc[i]*(-1))[0:100]
    fw.write('\n' + qrys[i] + ',')
    for s in rank:
        fw.write(docs[int(s)] + ' ')
fw.close()