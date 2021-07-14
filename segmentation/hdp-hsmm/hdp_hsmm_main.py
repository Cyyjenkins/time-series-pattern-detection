#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Yaoyu Chen
"""

import pyximport
pyximport.install()

import scipy.misc
import numpy as np
import pandas as pd

np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import copy, os
import re

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from openpyxl import Workbook

SAVE_FIGURES = False


def xlsx_save(data, path): # one sheet
    wb = Workbook()
    ws = wb.active
    [h, l] = data.shape  # h:rows, l:columns
    for i in range(h):
        row = []
        for j in range(l):
            row.append(data[i,j])
        ws.append(row)
    wb.save(path)


def obs_segment(durations, data_shape):
    segments = [0]
    ind = 0
    for n in durations:
        pos = ind + n
        if pos >= data_shape-1: break
        segments.append(pos)
        ind = pos
    return np.array((segments + [data_shape-1]), dtype=int)



def hdp_hsmm_infer(data_path, output_path):
    
    # hyperparameters
    Nmax = 50
    resample_iter = 150
    trunc = 60
    alpha, gamma, init_state_concentration = 6., 6., 6.
    obs_dim = 6
    obs_hypparams = {'mu_0':np.zeros(obs_dim),
                    'sigma_0':np.eye(obs_dim),
                    'kappa_0':0.25,
                    'nu_0':obs_dim+2}
    dur_hypparams = {'alpha_0':2*30,
                      'beta_0':2}
    
    obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
    dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]
    
    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
            alpha=alpha, gamma=gamma,  # these can matter; see concentration-resampling.py
            init_state_concentration=init_state_concentration,  # pretty inconsequential
            obs_distns=obs_distns,
            dur_distns=dur_distns)
    
    
    # data reading
    data_path = '../../data/'
    output_path = '../../output/seg/'
    
    print('data reading...')
    files = os.listdir(data_path)
    
    data_shape = []
    for file in files:
        data = pd.read_excel(data_path+file, header=None).values
        posteriormodel.add_data(data, trunc=trunc)
        data_shape.append(data.shape[0])
        
    
    # learning
    print('model resampling...')
    for idx in progprint_xrange(resample_iter):
        posteriormodel.resample_model()
    
    
    # data saving
    for i in range(len(files)):
        segments = obs_segment(posteriormodel.durations[i], data_shape[i])
        hidden_state = posteriormodel.stateseqs_norep[i]
    
        xlsx_save(segments.reshape((-1,1)), output_path+file)
        xlsx_save(hidden_state.reshape((-1,1)), output_path+'hidden_state_'+file)



if __name__ == '__main__':
    data_path = '../../data/'
    output_path = '../../output/lda/'
     
    hdp_hsmm_infer(data_path, output_path)