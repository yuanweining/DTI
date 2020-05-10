# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:34:08 2019

@author: a
"""
import pandas as pd
import numpy as np
import random
def get_drugbank_data(file):
    initial = pd.read_csv(file)
    protein_ids,protein_sequences,drug_ids,drug_sequences = select_512(file)
    
    x0_ids = []
    x1_ids = []
    x0_sequences = []
    x1_sequences = []
    y_ids = []
    y_sequences = []
    y = []
    
    f_x0_ids = []
    f_x1_ids = []
    f_x0_sequences = []
    f_x1_sequences = []
    f_y_ids = []
    f_y_sequences = []
    f_y = []
    
    _x0_ids = []
    _x1_ids = []
    _x0_sequences = []
    _x1_sequences = []
    _y_ids = []
    _y_sequences = []
    _y = []
    
    t_x0_ids = []
    t_x1_ids = []
    t_x0_sequences = []
    t_x1_sequences = []
    t_y_ids = []
    t_y_sequences = []
    t_y = []
    
    for i in range(len(protein_ids)):
        for j in range(i+1,len(protein_ids)):
            if(protein_ids[i] == protein_ids[j]):
                break;
            if(drug_ids[i] == drug_ids[j]):
                t_y_ids.append(drug_ids[i])
                t_y_sequences.append(drug_sequences[i])
                t_y.append(1)
                t_x0_ids.append(protein_ids[i])
                t_x1_ids.append(protein_ids[j])
                t_x0_sequences.append(protein_sequences[i])
                t_x1_sequences.append(protein_sequences[j])
            else:
                f_x0_ids.append(protein_ids[i])
                f_x1_ids.append(protein_ids[j])
                f_x0_sequences.append(protein_sequences[i])
                f_x1_sequences.append(protein_sequences[j])
                f_y_ids.append(0)
                f_y_sequences.append(0)
                f_y.append(0)
        
        
        
        
    
    rand_idx_x = np.random.randint(0,len(t_x0_ids),int(len(t_x0_ids)))
    rand_idx = np.random.randint(0,len(f_y_ids),len(rand_idx_x))
    
    for i in rand_idx_x:
        x0_sequences.append(t_x0_sequences[i])
        x1_sequences.append(t_x1_sequences[i])
        y.append(1)
    
    for i in rand_idx:
        _x0_ids.append(f_x0_ids[i])
        _x1_ids.append(f_x1_ids[i])
        _x0_sequences.append(f_x0_sequences[i])
        _x1_sequences.append(f_x1_sequences[i])
        _y_ids.append(0)
        _y.append(0)
        _y_sequences.append(0)
        
    
    
    x0_sequences = x0_sequences + _x0_sequences
    x1_sequences = x1_sequences + _x1_sequences
    y = y + _y
    ran_zip = list(zip(x0_sequences,x1_sequences,y))
    random.shuffle(ran_zip)
    x0_sequences[:], x1_sequences[:], y[:] = zip(*ran_zip)
    train_len = int(4*len(y)/5)
    
    return [x0_sequences[0:train_len],x1_sequences[0:train_len],y[0:train_len],x0_sequences[train_len:-1],x1_sequences[train_len:-1],y[train_len:-1]]
    
    '''
    true_data = {'protein0_ids':x0_ids,'protein1_ids':x1_ids,'protein0_sequences':x0_sequences,'protein1_sequences':x1_sequences,'y_ids':y_ids,'y_sequences':y_sequences,'y':y}
    fause_data = {'protein0_ids':_x0_ids,'protein1_ids':_x1_ids,'protein0_sequences':_x0_sequences,'protein1_sequences':_x1_sequences,'y_ids':_y_ids,'y_sequences':_y_sequences,'y':_y}
    '''
    
def select():
    file = 'F:/protein-sequence/protein-sequence-embedding-iclr2019-master/data/final.csv'
    initial = pd.read_csv(file)
    protein_sequences = list(initial.iloc[:,2])
    print('len : '+str(len(protein_sequences)))
    len_arr = []
    for i in protein_sequences:
        len_arr.append(len(i))
    len_arr.sort()
    print(len_arr[11500])
    

def select_512(file):
    initial = pd.read_csv(file)
    protein_ids = list(initial.iloc[:,1])
    protein_sequences = list(initial.iloc[:,2])
    drug_ids = list(initial.iloc[:,3])
    drug_sequences = list(initial.iloc[:,4])
    len_arr = []
    for i in protein_sequences:
        len_arr.append(len(i))
    sort_zip = list(zip(len_arr,protein_ids,protein_sequences,drug_ids,drug_sequences))
    sort_zip.sort()
    len_arr[:],protein_ids[:],protein_sequences[:],drug_ids[:],drug_sequences[:] = zip(*sort_zip)
    return [protein_ids[0:len_arr.index(510)],protein_sequences[0:len_arr.index(510)],drug_ids[0:len_arr.index(510)],drug_sequences[0:len_arr.index(510)]]

    
    

