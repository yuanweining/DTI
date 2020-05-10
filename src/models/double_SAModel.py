# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:15:28 2020

@author: a
"""

from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from src.utils import pack_sequences, unpack_sequences, unpack_torchtensor, pad_enc
import numpy as np
import math

class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 1, -float('inf'))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn
    
class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        out = q + residual
        return out, attn
    

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        return enc_output, enc_slf_attn


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Model(nn.Module):
    def __init__(self, nin, embedding_dim, hidden_dim, nlayers=1, dropout=0.1,
                 rnn_type='lstm', attn_dim = 64, nhead = 16):
        super(Model, self).__init__()
        self.alphabet = nin
        padding_idx = nin
        self.embed = nn.Embedding(nin+2, embedding_dim, padding_idx=padding_idx)
        
        #if rnn_type == 'lstm':
        #    RNN = nn.LSTM
        #elif rnn_type == 'gru':
        #    RNN = nn.GRU
        self.FFN_dim = int(hidden_dim/2)
        self.position = PositionalEmbedding(d_model=embedding_dim)
        self.segment = nn.Embedding(3, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        #self.encode_rnn = RNN(embedding_dim, hidden_dim, nlayers, batch_first=True
        #              , bidirectional=True, dropout=dropout)
        self.attention = EncoderLayer(hidden_dim, nhead, attn_dim, attn_dim*2)
        self.link_attention = EncoderLayer(hidden_dim, nhead, attn_dim, attn_dim*2)
        self.alpha = 10
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.projection1 = nn.Linear(hidden_dim, self.FFN_dim)
        self.GELU = GELU()
        self.projection2 = nn.Linear(self.FFN_dim, hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        
    def encode(self, X):
        x = X[0]
        lengths = X[1]
        h = self.embed(x) + self.position(x) 
        enc_masks = self.generate_sent_masks(h, lengths)
        attn = self.input_projection(h)
        for i in range(2):
            attn, attn_weights = self.attention(attn, enc_masks)
            res = attn
            attn = self.layer_norm(attn)
            attn = self.projection2(self.dropout(self.GELU(self.projection1(attn))))
            attn = res + self.dropout(attn)
            
        out = []
        for i in range(len(lengths)):
            out.append(attn[i,:lengths[i],:])
        
        return out, attn_weights
    
    def link(self, x0, x1, end_token):
        # x0 = x1 = [len_seq, hidden_size] * batchsize
        b = len(x0)
        separator = torch.from_numpy(np.array([end_token]*b)).long().unsqueeze(1).cuda()
        separator = self.embed(separator)
        separator = self.input_projection(separator)
        x, lengths = pad_enc(x0, x1, separator)
        link_masks = self.generate_sent_masks(x, lengths)
        
        attn, attn_weights = self.link_attention(x, link_masks)
        res = attn
        attn = self.layer_norm(attn)
        attn = self.projection2(self.dropout(self.GELU(self.projection1(attn))))
        attn = res + self.dropout(attn)
        
        out = torch.zeros(b,1,attn.shape[2]).cuda()
        for i in range(len(lengths)):
            out[i,:,:] = attn[i,x0[i].shape[0],:]
        out = self.sigmoid(self.out_projection(out.squeeze())).squeeze()
        
        
        
        return out
        
    def get_seg(self, x):
        b = int(x.shape[0]/2)
        seg = (x!=self.alphabet).long()
        seg[b:, :] = 2*seg[b:, :]
        return seg
        
        
    
    def generate_sent_masks(self, enc_hiddens, source_lengths):
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        enc_masks = enc_masks.unsqueeze(1).expand(-1, enc_hiddens.size(1), -1)    
        return enc_masks.cuda()
    
    def L1(self, x, y):
        return -torch.sum(torch.abs(x.unsqueeze(1)-y), -1)
    
    def score(self, c0, c1, align_method):
        if align_method == 'me':
            c = torch.zeros(len(c0)).cuda()
            for i in range(len(c0)):
                s = self.L1(c0[i], c1[i])
                c[i] = torch.mean(s) + self.alpha 
            m = nn.Sigmoid()
            c = m(c)
        
        if align_method == 'ya':
            c = torch.zeros(len(c0)).cuda()
            for i in range(len(c0)):
                z_x = c0[i]
                z_y = c1[i]
                s = self.L1(z_x, z_y)
                a = torch.mean(torch.abs(z_x),-1)
                b = torch.mean(torch.abs(z_y),-1)
                a = F.softmax(a,0)
                b = F.softmax(b,0)
                a = a.unsqueeze(1)
                b = b.unsqueeze(0)
                a = a.expand(a.shape[0],b.shape[1])
                b = b.expand(a.shape[0],b.shape[1])
                k = a*b
                score = torch.sum(k*s)/torch.sum(k)
                c[i] = score + self.alpha
            m = nn.Sigmoid()
            c = m(c)
            
        return c