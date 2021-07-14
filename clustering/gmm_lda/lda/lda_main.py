# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:54:25 2019

@author: Yaoyu Chen
"""

import copy
import pylab
import random

import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

from openpyxl import Workbook


class LDA():
    
    def __init__(self, data, k, iterations, num_word, alpha=1, beta=1):      
        self.__data = data
        self.__num_doc = len(data)
        self.__num_word = num_word
        self.__k = k
        self.__iter = iterations
        self.__alpha = alpha
        self.__beta = beta
        
        dw_count = np.zeros((len(data), num_word), dtype=int)
        for i in range(len(data)):
            for j in range(data[i].shape[0]):
                dw_count[i,data[i][j]] += 1
        self.__doc_word = dw_count
    
    
    def show_document_word(self, data):
        L = len(data)
        doc = []
        for l in range(L):
            for n in range(data[l]):
                doc.append(l)
        return doc
    
    
    def number_account(self, doc_word, doc_topic):        
        n_dz = np.zeros((self.__num_doc, self.__k))  # number of document_topic
        n_zw = np.zeros((self.__k, self.__num_word)) # number of topic_word
        n_z  = np.zeros((self.__k, 1))               # number of topic
        
        for d in range(self.__num_doc):
            N = len(doc_word[d])
            for n in range(N):
                w = doc_word[d][n]
                z = doc_topic[d][n]
                n_dz[d][z] += 1
                n_zw[z][w] += 1
                n_z[z] += 1
        
        n_dz = n_dz.astype(np.int)
        n_zw = n_zw.astype(np.int)
        n_z  = n_z.astype(np.int)
        return n_dz, n_zw, n_z
    
    
    def multinomial_sampling(self, d, w, n_dz, n_zw, n_z):
        P = [ 0.0 ] * self.__k
        P = ((n_dz[d,:] + self.__alpha )*(n_zw[:,w] + self.__beta)).T \
        /(n_z[:] + self.__num_word*self.__beta).reshape(self.__k,)
        
        for z in range(1,self.__k):
            P[z] = P[z] + P[z-1]
    
        rnd = P[self.__k - 1] * random.random()
        for z in range(self.__k):
            if P[z] >= rnd:
                return z
    
    
    def likelihood_calculation(self, n_dz, n_zw, n_z):
        lik = 0
        Pw_z = (n_zw.T + self.__beta) / (n_z.T + self.__num_word *self.__beta)
        for d in range(len(self.__data)):
            Pz = (n_dz[d] + self.__alpha) / (np.sum(n_dz[d]) \
                  + self.__k*self.__alpha)
            Pwz = Pz * Pw_z
            Pw = np.sum( Pwz , 1 ) + 1e-5
            lik += np.sum( self.__doc_word[d] * np.log(Pw) )             
        return lik
    
    
    def plot(self, it, liks):
        print('iter:%s'%it, " log-like:%.2f"%liks[-1])

    
    
    def lda_infer(self):
        pylab.ion()
        # save the likelihood of each iterations
        liks = []        
        
        # show the document_word matrix and document_topic matrix
        doc_word = copy.deepcopy(self.__data)
        doc_topic = [ None for i in range(self.__num_doc) ]
        
        for d in range(self.__num_doc):
            doc_topic[d] = np.random.randint(0, self.__k, len(doc_word[d])).tolist()
        
        # account for the number of doc_topic, topic_word and topics
        n_dz, n_zw, n_z = self.number_account(doc_word, doc_topic)
                
        for it in range(self.__iter):
            for d in range(self.__num_doc):
                N = len(doc_word[d])
                for n in range(N):
                    w = doc_word[d][n]
                    z = doc_topic[d][n]            
        
                    n_dz[d][z] -= 1
                    n_zw[z][w] -= 1
                    n_z[z] -= 1
                    
                    # sampling according to coditional probability
                    z = self.multinomial_sampling(d, w, n_dz, n_zw, n_z)
        
                    doc_topic[d][n] = z
                    n_dz[d][z] += 1
                    n_zw[z][w] += 1
                    n_z[z] += 1
        
            lik = 0
            lik = self.likelihood_calculation(n_dz, n_zw, n_z)
            liks.append(lik)
            self.plot(it, liks)
        
        return n_dz, n_zw, n_z, doc_topic



        
        
