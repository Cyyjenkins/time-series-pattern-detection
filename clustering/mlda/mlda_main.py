# -*- coding: utf-8 -*-
'''# -*- coding: utf-8-sig -*-'''
"""
Created on Sat Apr 13 16:18:01 2019

@author: Jenkins
"""
# encoding: shift_jis
import re
import numpy
import random
import pylab
import pickle
import os
import time

import numpy as np
import pandas as pd
import multiprocessing as mp
from openpyxl import Workbook


def calc_lda_param(docs_mdn, topics_mdn, K, dims):
    M = len(docs_mdn)     #       number of modality
    D = len(docs_mdn[0])  #       number of documents
                          # K:    number of topics
                          # dims: dimention of modality

    n_mdz = numpy.zeros((M,D,K), dtype=int)
    n_dz = numpy.zeros((D,K), dtype=int)
    n_mzw = [ numpy.zeros((K,dims[m]), dtype=int) for m in range(M)]
    n_mz = [ numpy.zeros(K, dtype=int) for m in range(M) ]

    for d in range(D):
        for m in range(M):
            if dims[m]==0:
                continue
            
            N = len(docs_mdn[m][d])
            for n in range(N):
                w = docs_mdn[m][d][n]
                z = topics_mdn[m][d][n]
                
                n_mdz[m][d][z] += 1    # number counting
                n_mzw[m][z][w] += 1
                n_mz[m][z] += 1
                
    n_dz = np.sum(n_mdz, axis=0)
    return n_dz, n_mzw, n_mz, n_mdz


# Gibbs Sampling
def sample_topic(d, w, n_dz, n_zw, n_z, K, V):
    __alpha, __beta = 1.0, 1.0 
    P = [0.0]*K
    P = (n_dz[d,:] + __alpha) * (n_zw[:,w] + __beta) / (n_z[:] + V*__beta)

    for z in range(1,K):
        P[z] = P[z] + P[z-1]

    rnd = P[K-1] * random.random()
    for z in range(K):
        if P[z] >= rnd:
            return z


def conv_to_word_list( data ):
    V = len(data)
    doc = []
    for v in range(V):
        for n in range(data[v]):
            doc.append(v)
    return doc


# calculate the likelihood of the appearance of each word
# and then calculate the overall likelihood
def calc_liklihood(data, n_dz, n_zw, n_z, K, V):
    lik, __alpha, __beta = 0, 1., 1.
    # topic-word proportions
    P_wz = (n_zw.T + __beta) / (n_z + V *__beta)
    for d in range(len(data)):
        # doc-topic proportions
        Pz = (n_dz[d] + __alpha)/(numpy.sum(n_dz[d]) + K*__alpha)
        Pwz = Pz * P_wz
        Pw = numpy.sum(Pwz,1) + 1e-5
        lik += numpy.sum(data[d] * numpy.log(Pw))
    return lik


def save_model(save_dir, n_mdz, n_mzw):
    try: os.makedirs(save_dir)
    except: pass

    with open(os.path.join(save_dir, "model.pickle"), "wb") as f:
        pickle.dump([n_mdz, n_mzw], f)


def load_model( load_dir ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open(model_path, "rb" ) as f:
        a,b = pickle.load( f )
    return a,b


def xlsx_save(data,path):  # one sheet
    wb = Workbook()    
    ws = wb.active   
    [h, l] = data.shape  # h:rows，l:columns
    for i in range(h):
        row = []
        for j in range(l):
            row.append(data[i,j])
        ws.append(row)
    wb.save(path)
    
    
def mlda( data, docs_mdn, K, num_itr=100, save_dir="model", load_dir=None ):
    liks = []
    M = len(data)
    dims = []
    for m in range(M):
        if data[m] is not None:
            dims.append(len(data[m][0]))
            D = len(data[m])
        else: dims.append(0)

    topics_mdn = [[ None for i in range(D) ] for m in range(M)]
    for d in range(D):
         for m in range(M):
            if data[m] is not None:
                topics_mdn[m][d] = numpy.random.randint(0,K,len(docs_mdn[m][d]))
                                
    n_dz, n_mzw, n_mz, n_mdz = calc_lda_param( docs_mdn, topics_mdn, K, dims )

    if load_dir:
        n_mzw, n_mz = load_model( load_dir )

    for it in range(num_itr):
        if it%50 == 0:
            print('iterations:%s'%(it))

        for d in range(D):
            for m in range(M):
                if data[m] is None:
                    continue

                N = len(docs_mdn[m][d])
                for n in range(N):
                    w = docs_mdn[m][d][n]
                    z = topics_mdn[m][d][n]

                    n_dz[d][z] -= 1
                    if not load_dir:
                        n_mzw[m][z][w] -= 1
                        n_mz[m][z] -= 1
                        n_mdz[m][d][z] -= 1

                    z = sample_topic(d,w,n_dz,n_mzw[m],n_mz[m],K,dims[m])

                    topics_mdn[m][d][n] = z
                    n_dz[d][z] += 1

                    if not load_dir:
                        n_mzw[m][z][w] += 1
                        n_mz[m][z] += 1
                        n_mdz[m][d][z] += 1

        lik = 0
        for m in range(M):
            if data[m] is not None:
                lik += calc_liklihood(data[m],n_dz,n_mzw[m],n_mz[m],K,dims[m])
        liks.append(lik)

    return n_dz, n_mzw, n_mz, n_mdz, liks, topics_mdn


def read_data(data_path, seg_path):
    # input data and segmentation results
    # each segment is considered as a document 
    Y, seg_list = np.zeros((0,0)), []
    for file in os.listdir(data_path):
        data = pd.read_excel(data_path+file, header=None).values
        Y = data if not Y.shape[0] else np.vstack((Y, data))
            
        seg_pos = pd.read_excel(seg_path+file, header=None).values.flatten()
        seg_num = (seg_pos[1:] - seg_pos[:-1]).tolist()
        seg_num[0] += 1
        seg_list += seg_num
    return Y, seg_list
   

def convert_document_shape(Y, seg_list, word_num=50):
    # discretization
    Y = (Y-Y.min(0))/(Y.max(0)-Y.min(0))  # normalization
    Y = (Y*(word_num-1)).astype(np.int)
     
    # counting of words for each document in each dimension
    dim = Y.shape[1]
    doc_num = len(seg_list)
    doc_word_cnt = []
    for d in range(dim):
        doc_word = np.zeros((doc_num,word_num), dtype=int)
        ind = 0
        for i,n in enumerate(seg_list):
            Y0 = Y[ind:ind+n,d]
            ind += n
            for j in range(word_num):
                doc_word[i,j] = len([k for k in Y0 if k == j])
        doc_word_cnt.append(doc_word) 

    # word order for each document in each dimension
    ord_data = []
    for d in range(dim):
        ind = 0
        o_data_tmp = []
        for i,n in enumerate(seg_list):
            o_data_tmp.append(Y[ind:ind+n,d])
            ind += n
        ord_data.append(o_data_tmp)
          
    return doc_word_cnt, seg_list, Y, ord_data


def convert_label(topics_mdn,seg_list):
    dim = len(topics_mdn)
    label = np.zeros((0,0)) 
    for d in range(dim):
        w_patt = np.zeros((0,0))
        for dim_patt in topics_mdn[d]:
            w_patt = dim_patt if w_patt.shape[0]==0 else np.hstack((w_patt,dim_patt))
        label = w_patt if label.shape[0]==0 else np.vstack((label,w_patt))
    label = label.T
    
    doc_label = np.zeros((len(seg_list),), dtype=int)
    ind = 0
    for i,n in enumerate(seg_list):
        doc_label[i] = np.argmax(np.bincount(label[ind:ind+n].flatten()))
        ind += n
    word_label = np.argmax(label,axis=1)
    return label, doc_label, word_label


def mlda_infer(data_path, seg_path, output_path):

    running_time = time.perf_counter()
    
    # hyper-parameters
    word_num = 20
    iter_num = 100
    patt_num = 5
    
    # data reading
    Y, seg_list = read_data(data_path, seg_path)
    doc_word_cnt,seg_list,Y,ord_data = convert_document_shape(Y,seg_list,word_num)

    # mlda inferring
    n_dz, n_mzw, n_mz, n_mdz, liks, topics_mdn = mlda(doc_word_cnt, ord_data, 
                                                      patt_num, iter_num, 
                                                      "learn_result")

    # find word label and document label
    label, doc_label, word_label = convert_label(topics_mdn, seg_list)

    # saving
    save_model(output_path, n_mdz, n_mzw)
    xlsx_save(doc_label.reshape((-1,1)), output_path+'doc_label.xlsx')
    xlsx_save(word_label.reshape((-1,1)), output_path+'word_label.xlsx')
    
    running_time = (time.perf_counter() - running_time)
    print("Time used:%.4ss"%(running_time))
        

if __name__ == '__main__':
    data_path = '../../data/'
    seg_path = '../../output/seg/'
    output_path = '../../output/lda/'
    
    mlda_infer(data_path, seg_path, output_path)
