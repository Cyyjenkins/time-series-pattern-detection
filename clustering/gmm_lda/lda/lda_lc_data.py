# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:30:06 2019

@author: Jenkins
"""

import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

#from ts_lda import ts_lda
from ord_ts_lda import ts_lda
from openpyxl import Workbook



def ord_doc_word(data):
    doc_word = []
    for i in range(data.shape[0]):
        doc_word.append(data[i,1:data[i,0]+1])
    return doc_word


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
    

def create_folder(path):
    sp_folder = path.split('/')
    folder = sp_folder[0]
    for i in range(len(sp_folder)-1):
        if not os.path.exists(folder): os.mkdir(folder)
        folder = folder + '/' + sp_folder[i+1]


'''ordered version'''
def lda_lc_data(pat_num, dset):
    word = 50
    
    """bmass and bmoss"""
    year = 14
    # for year in [14,15]:
    doc_word = ord_doc_word(pd.read_excel('../GMM/src/results/%s_set/GA_%s_doc.xlsx'%(dset,year),header=None).values)
    
    obj = ts_lda(doc_word, pat_num, 200, word, alpha=1, beta=1)
    (n_dz, n_zw, n_z, doc_word, doc_topic) = obj.lda_infer()
    
    # computing word label
    word_label = np.array(doc_topic[0])
    for i in range(1,len(doc_topic)):
        word_label = np.hstack((word_label,np.array(doc_topic[i])))
    word_label = word_label.reshape((word_label.shape[0],1)).astype(int)
        
    # computing document label
    doc_label = np.zeros((n_dz.shape[0],1), dtype=int)
    for i in range(n_dz.shape[0]):
        doc_label[i,0] = np.argmax(n_dz[i,:])
    
    if dset == 'train':
        xlsx_save(n_dz,'results/%s/GA,20%s/doc_topic.xlsx'%(pat_num,year))
        xlsx_save(n_z,'results/%s/GA,20%s/topic.xlsx'%(pat_num,year))
        xlsx_save(n_zw,'results/%s/GA,20%s/topic_word.xlsx'%(pat_num,year))
        xlsx_save(word_label,'results/%s/GA,20%s/word_label.xlsx'%(pat_num,year))
        xlsx_save(doc_label,'results/%s/GA,20%s/doc_label.xlsx'%(pat_num,year))
    if dset == 'test':
        xlsx_save(n_dz,'results/test/GA,20%s/doc_topic.xlsx'%(year))
        xlsx_save(n_z,'results/test/GA,20%s/topic.xlsx'%(year))
        xlsx_save(n_zw,'results/test/GA,20%s/topic_word.xlsx'%(year))
        xlsx_save(word_label,'results/test/GA,20%s/word_label.xlsx'%(year))
        xlsx_save(doc_label,'results/test/GA,20%s/doc_label.xlsx'%(year))            
    print('%s pattern complete...'%pat_num)

    """hdp-hsmm"""
    # # doc_word = ord_doc_word(pd.read_excel('../GMM/src/results/%s_set/filter_hdp_hsmm_doc.xlsx'%(dset),header=None).values)
    # doc_word = ord_doc_word(pd.read_excel('../GMM/src/results/%s_set/hdp_hsmm_doc.xlsx'%(dset), header=None).values)
        
    # obj = ts_lda(doc_word, pat_num, 200, word, alpha=1, beta=1)
    # (n_dz, n_zw, n_z, doc_word, doc_topic) = obj.lda_infer()
    
    # # computing word label
    # word_label = np.array(doc_topic[0])
    # for i in range(1,len(doc_topic)):
    #     word_label = np.hstack((word_label,np.array(doc_topic[i])))
    # word_label = word_label.reshape((word_label.shape[0],1)).astype(int)
        
    # # computing document label
    # doc_label = np.zeros((n_dz.shape[0],1), dtype=int)
    # for i in range(n_dz.shape[0]):
    #     doc_label[i,0] = np.argmax(n_dz[i,:])
    
    # # data saving
    # if dset == 'train':
    #     path = 'results/%s/nf_hdp_hsmm/'%pat_num
    # elif dset == 'test':
    #     path = 'results/test/hdp_hsmm/'
        
    # create_folder(path)
    # xlsx_save(n_dz, path+'doc_topic.xlsx')
    # xlsx_save(n_z, path+'topic.xlsx')
    # xlsx_save(n_zw, path+'topic_word.xlsx')
    # xlsx_save(word_label, path+'word_label.xlsx')
    # xlsx_save(doc_label, path+'doc_label.xlsx')  
          
    # print('%s pattern complete...'%pat_num)
    
    
"""multi Processing"""
#if __name__ == '__main__':
#    print('start lda inferring')
#    Proc = []  
#    pat_num = [_ for _ in range(3,11)]
#    
#    for i in range(len(pat_num)):
#        p = mp.Process(target=lda_lc_data, args=(pat_num[i],))
#        p.start()
#        Proc.append(p)
#    for pro in Proc:
#        pro.join()                   
#    print('lda inferring complete!')

#for patt_num in range(4,11):
#    lda_lc_data(patt_num, 'train')

"""test"""
running_time = time.perf_counter()

#for i in range(3, 11):
#    print('i == %s'%(i))
#    lda_lc_data(i, 'train')
lda_lc_data(5, 'test')

running_time = (time.perf_counter() - running_time)
print("Time used: %.6ss"%(running_time))


