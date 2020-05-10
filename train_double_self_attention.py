# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:56:32 2019

@author: a
"""

from __future__ import print_function,division

import numpy as np
import pandas as pd
import sys

from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch.utils.data
from src.alphabets import Uniprot21
from src.utils import pack_sequences, unpack_sequences, pad_sents
from src.utils import PairedDataset, collate_paired_sequences
from src.drugbank import get_drugbank_data
import src.models.double_SAModel

'''
def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
'''

def main():
    import argparse
    parser = argparse.ArgumentParser('Script for training embedding model on SCOP.')

    parser.add_argument('--dev', action='store_true', help='use train/dev split')

    parser.add_argument('-m', '--model', choices=['ssa', 'ua', 'me'], default='ssa', help='alignment scoring method for comparing sequences in embedding space [ssa: soft symmetric alignment, ua: uniform alignment, me: mean embedding] (default: ssa)')
    parser.add_argument('--allow-insert', action='store_true', help='model insertions (default: false)')

    parser.add_argument('--norm', choices=['l1', 'l2'], default='l1', help='comparison norm (default: l1)')

    parser.add_argument('--rnn-type', choices=['lstm', 'gru'], default='lstm', help='type of RNN block to use (default: lstm)')
    parser.add_argument('--input-dim', type=int, default=20, help='dimension of input to RNN (default: 512)')
    parser.add_argument('--rnn-dim', type=int, default=32, help='hidden units of RNNs (default: 256)')
    parser.add_argument('--num-layers', type=int, default=2, help='number of RNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability (default: 0)')
    parser.add_argument('--epoch-scale', type=int, default=5, help='scaling on epoch size (default: 5)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs (default: 100)')

    parser.add_argument('--batch-size', type=int, default=16, help='minibatch size (default: 64)')

    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--tau', type=float, default=0.5, help='sampling proportion exponent (default: 0.5)')
    parser.add_argument('--augment', type=float, default=0, help='probability of resampling amino acid for data augmentation (default: 0)')
    parser.add_argument('--lm', help='pretrained LM to use as initial embedding')

    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--alignmethod', type=str, default='me', help='align-mothod')
    parser.add_argument('--gpu', type=int, default='0', help='gpu')
    parser.add_argument('--head', type=int, default='4', help='head')
    parser.add_argument('--attndim', type=int, default='32', help='attn_dim')
    

    args = parser.parse_args()
    
    
    filename = 'SAout.txt'
    f=open(filename, 'w')
    
    align_method = args.alignmethod
    
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)


    alphabet = Uniprot21()

    print('#loading drugbank train data', file=f)
    _x0_train,_x1_train,_y_train,x0_test,x1_test,y_test = get_drugbank_data('data/final.csv')
    x0_train = [x.encode('utf-8').upper() for x in _x0_train]
    x0_train = [torch.from_numpy(alphabet.encode(x)).long() for x in x0_train]
    x1_train = [x.encode('utf-8').upper() for x in _x1_train]
    x1_train = [torch.from_numpy(alphabet.encode(x)).long() for x in x1_train]
    print('scare : '+str(len(x0_train)), file=f)

    
    y_train = np.array(_y_train)
    y_train = torch.from_numpy(y_train).long()
    print('x0_train[0].shape : ' + str(type(x0_train[0])), file=f)
    print('len of x0_train : '+str(len(x0_train[0])), file=f)
    print('y_train[0].shape : ' + str(y_train), file=f)
    
    
    print('# loading drugbank test data', file=f)
    x0_test = [x.encode('utf-8').upper() for x in x0_test]
    x0_test = [torch.from_numpy(alphabet.encode(x)).long() for x in x0_test]
    x1_test = [x.encode('utf-8').upper() for x in x1_test]
    x1_test = [torch.from_numpy(alphabet.encode(x)).long() for x in x1_test]
    
    
    y_test = np.array(y_test)
    y_test = torch.from_numpy(y_test).long()
    print('x0_test[0].shape : ' + str(type(x0_test[0])), file=f)
    print('len of x0_test : '+str(len(x0_test[0])), file=f)
    print('y_test[0].shape : ' + str(y_test), file=f)
    
    
    dataset_test = PairedDataset(x0_test, x1_test, y_test)
    
    dataset_train = PairedDataset(x0_train, x1_train, y_train)
    
    epoch_size = len(x0_train)
    epoch_size_test = len(x0_test)
    
    batch_size = args.batch_size
    head = args.head
    attn_dim = args.attndim
    

    train_iterator = torch.utils.data.DataLoader(dataset_train
                                                , batch_size=batch_size
                                                #, sampler=sampler
                                                , collate_fn=collate_paired_sequences
                                                )
    
    test_iterator = torch.utils.data.DataLoader(dataset_test
                                               , batch_size=batch_size
                                               , collate_fn=collate_paired_sequences
                                               )
    

    rnn_dim = args.rnn_dim
    num_layers = args.num_layers


    input_dim = args.input_dim
    dropout = args.dropout
    
    allow_insert = args.allow_insert

    print('# initializing model with:', file=f)
    print('# attention_size:', attn_dim, file=f)
    print('# input_dim:', input_dim, file=f)
    print('# rnn_dim:', rnn_dim, file=f)
    print('# num_layers:', num_layers, file=f)
    print('# dropout:', dropout, file=f)
    print('# allow_insert:', allow_insert, file=f)



    lm = None
    if args.lm is not None:
        lm = torch.load(args.lm)
        lm.eval()
        for param in lm.parameters():
            param.requires_grad = False
        print('# using LM:', args.lm, file=f)

    model = src.models.double_SAModel.Model(len(alphabet), input_dim, rnn_dim
                                                   , nlayers=num_layers, dropout=dropout, attn_dim = attn_dim, nhead = head)

    #model.apply(weights_init)
    #model = torch.nn.DataParallel(model)

    '''
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    input()
    '''
    

    if use_cuda:
        model.cuda()

    num_epochs = args.num_epochs

    weight_decay = args.weight_decay
    lr = args.lr

    print('# training with Adam: lr={}, weight_decay={}'.format(lr, weight_decay), file=f)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    print('# training model', file=f)

    output = args.output
    if output is None:
        output = sys.stdout
    else:
        output = open(output, 'w')
    digits = int(np.floor(np.log10(num_epochs))) + 1
    line = '\t'.join(['epoch', 'split', 'loss', 'mse', 'accuracy', 'r', 'rho' ])
    print(line, file=f)
    f.close()
    
    
    

    for epoch in range(num_epochs):
        # train epoch
        model.train()
        n = 0
        loss_estimate = 0
        
        for x0,x1,y in train_iterator:
            
            
            if use_cuda:
                y = y.cuda()
            y = Variable(y)
            
            b = len(x0)  
            x = x0 + x1
            lengths = np.array([len(_x) for _x in x])
            pad_token = len(alphabet)
            x = pad_sents(x, pad_token, return_tensor = True)
            x.required_grad = True
            enc_out, attn_weight = model.encode([x, lengths]) 
            enc_out0 = enc_out[:b]
            enc_out1 = enc_out[b:]
            
            end_token = len(alphabet)+1
            out = model.link(enc_out0, enc_out1, end_token)
            
            
            loss_func = nn.BCELoss()
            loss = loss_func(out, y.float())
            loss.backward()

            optim.step()
            optim.zero_grad()

            acc = (y.float().eq((out+0.5).int().float()).sum().item())/b
            
            n += b
            delta = b*(loss.item() - loss_estimate)
            loss_estimate += delta/n
            if (n - b)//100 < n//100:
                with open(filename, 'a') as f:
                    print('# [{}/{}] training {:.1%} loss={:.5f}, acc={:.5f}\n'.format(epoch+1
                                                                    , num_epochs
                                                                    , n/epoch_size
                                                                    , loss 
                                                                    , acc
                                                                    )
                         , end='\r', file=f)
        output.flush()

        # eval and save model
        model.eval()

        
        n = 0
        with torch.no_grad():
            for x0,x1,y in test_iterator:

                if use_cuda:
                    y = y.cuda()
                y = Variable(y)
                
                b = len(x0)  
                x = x0 + x1
                lengths = np.array([len(_x) for _x in x])
                pad_token = len(alphabet)
                x = pad_sents(x, pad_token, return_tensor = True)
                x.required_grad = True
                enc_out, attn_weight = model.encode([x, lengths]) 
                enc_out0 = enc_out[:b]
                enc_out1 = enc_out[b:]
                
                end_token = len(alphabet)+1
                out = model.link(enc_out0, enc_out1, end_token)
            
            
                loss_func = nn.BCELoss()
                loss = loss_func(out, y.float())

                acc = (y.float().eq((out+0.5).int().float()).sum().item())/b
                
                n += b
                delta = b*(loss.item() - loss_estimate)
                loss_estimate += delta/n
                if (n - b)//100 < n//100:
                    with open(filename, 'a') as f:
                        print('# [{}/{}] testing {:.1%} loss={:.5f}, acc={:.5f}\n'.format(epoch+1
                                                                        , num_epochs
                                                                        , n/epoch_size_test
                                                                        , loss 
                                                                        , acc
                                                                        )
                             , end='\r', file=f)
        output.flush()
        
        
        save_path = 'SA_pre_model' + '_epoch' + str(epoch+1).zfill(digits) + '.pkl'
        model.cpu()
        torch.save(model.state_dict(), save_path)
        if use_cuda:
            model.cuda()


if __name__ == '__main__':
    main()
    





