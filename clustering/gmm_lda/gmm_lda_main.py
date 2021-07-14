#! /usr/bin/env python

import sys 
sys.path.append("../../utils/")
sys.path.append("gmm/")
sys.path.append("lda/")


import re
import os
import time
import pickle

import pylab as pl
import numpy as np
import pandas as pd
import multiprocessing as mp

from gmm.gmm_main import GMM
from gmm.normal import Normal
from lda_main import LDA

import files_operations as fo


def gmmlda_read_data(data_path, seg_path):
    Y, seg_list = np.zeros((0,0)), []
    for file in os.listdir(data_path):
        data = pd.read_excel(data_path+file, header=None).values
        Y = data if not Y.shape[0] else np.vstack((Y, data))
            
        seg_pos = pd.read_excel(seg_path+file, header=None).values.flatten()
        seg_num = (seg_pos[1:] - seg_pos[:-1]).tolist()
        seg_num[0] += 1
        seg_list += seg_num
    return Y, seg_list

    
def word_2_doc(word_label, seg_list):
    doc, ind = [], 0
    for n in seg_list:
        doc.append(word_label[ind:ind+n])
        ind += n
    return doc


def doc_2_topic(doc_topic):
    doc_label = np.array([max(n, key=n.count) for n in doc_topic])
    word_label = []
    for n in doc_topic:
        word_label += n
    word_label = np.array(word_label)
    return doc_label, word_label


def save_model(save_dir, n_mdz, n_mzw):
    try: os.makedirs(save_dir)
    except: pass

    with open(os.path.join(save_dir, "model.pickle"), "wb") as f:
        pickle.dump([n_mdz, n_mzw], f)
        

def gmmlda_infer(data_path, seg_path, output_path):
    # hyper-parameters
    word_num = 20
    gmm_iter = 10 
    lda_iter = 10
    patt_num = 5
    
    # read data
    Y, seg_list = gmmlda_read_data(data_path, seg_path)
    
    # gmm
    print('gmm')
    gmm = GMM(dim=Y.shape[1], ncomps=word_num, data=Y, method="kmeans")  
    word, prob = gmm.em(Y, nsteps=gmm_iter)
    
    fo.xlsx_save(word.reshape((word.shape[0],1)),
                 output_path + 'gmm_res.xlsx')
    
    # lda
    print('lda')
    doc = word_2_doc(word, seg_list)
        
    obj = LDA(doc, patt_num, lda_iter, word_num, alpha=1, beta=1)
    n_dz, n_zw, n_z, doc_topic = obj.lda_infer()
    
    # save data
    save_model(output_path, n_dz, n_zw)
    doc_label, word_label = doc_2_topic(doc_topic)
    fo.xlsx_save(doc_label, output_path+'doc_label.xlsx')
    fo.xlsx_save(word_label, output_path+'word_label.xlsx')



if __name__ == '__main__':
    data_path = '../../data/'
    seg_path = '../../output/seg/'
    output_path = '../../output/lda/'
    
    gmmlda_infer(data_path, seg_path, output_path)