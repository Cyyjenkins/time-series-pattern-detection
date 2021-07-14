# -*- coding: utf-8 -*-


import yaml
import os

import numpy as np
import pandas as pd

from copy import deepcopy
from numpy import random
from numpy import linalg
from dateutil import parser as dp
from change_detec import Bcdm
from change_detec import MatrixVariateNormalInvGamma
from openpyxl import Workbook



def find_data_dim(data_path):
    return pd.read_excel(data_path+os.listdir(data_path)[0], header=0).shape[1]


def yaml_parse(yaml_path, dim):
    with open(yaml_path) as f:
        hyper_params = yaml.load(f, Loader=yaml.FullLoader)
    return dict(((key,eval(value) if isinstance(value,str) else value) for key,value in hyper_params.items()))


def xlsx_save(data, path, header=False):
    try:os.makedirs(path)
    except:pass
    
    if len(data.shape) == 1: data = data[:, np.newaxis]
    wb = Workbook()
    ws = wb.active
    if header: ws.append(header)
    [h, l] = data.shape  # h:rows，l:columns
    for i in range(h):
        row = []
        for j in range(l):
            row.append(data[i,j])
        ws.append(row)
    wb.save(path)


def bmoss_infer(data_path, output_path):
    dim = find_data_dim(data_path)
    # hyper_params = yaml_parse('hyper_params.yaml', dim)
    
    # hyper_params
    hyper_params = {
        'omega': 1.0e-3 * np.eye(1),
        'sigma': 1.0e-6 * np.eye(dim),
        'lamb': 0.05}
    

    for file in os.listdir(data_path):

        bcdm_probabilities = Bcdm(alg='sumprod', ratefun=hyper_params['lamb'],
                                  omega=deepcopy(hyper_params['omega']), 
                                  sigma=deepcopy(hyper_params['sigma']))
    
        bcdm_segments = Bcdm(alg='maxprod', ratefun=hyper_params['lamb'],
                             omega=deepcopy(hyper_params['omega']), 
                             sigma=deepcopy(hyper_params['sigma']))
        
        
        Y = pd.read_excel(data_path+file, header=None).values
        X = np.array(range(Y.shape[0]), dtype=int)
        
        for x, y in zip(X, Y):
            y = np.array([y])
            bcdm_probabilities.update(x, y)
            bcdm_segments.update(x, y)

        hypotheses_probability = bcdm_probabilities.infer()
        segments = bcdm_segments.infer()

        # head and tail
        if segments[0] < 0: segments[0] = 0
        if segments[-1] != Y.shape[0]-1:
            segments.append(Y.shape[0]-1)
        xlsx_save(np.array(segments), output_path+file)



if __name__ == '__main__':
    data_path = '../../data/'
    output_path = '../../output/seg/'
    
    bmoss_infer(data_path, output_path)