# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:34:08 2019

@author: a
"""
import pandas as pd

def get_predict_data(file):
    initial = pd.read_csv(file)
    pdb_id = initial.iloc[:,0].values.tolist()
    sequence = initial.iloc[:,1].values.tolist()
    return pdb_id,sequence

def get_drug(id,file):
    initial = pd.read_csv(file)
    drugs = initial.iloc[:,-1]
    return [drugs[id]]
    
#get_drug(10,'../data/all.csv')
    
