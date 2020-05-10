from __future__ import print_function,division

import numpy as np

import torch
import torch.utils.data
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
#from matplotlib import pyplot as plt 
'''
def plot(weight):
    weight = weight.detach().cpu().numpy()
    a = np.max(weight, axis=1)
    print(a)
    weight = weight[0:6, 0:30]
    plt.imshow(weight)  
    plt.colorbar()
    plt.show() 
'''   

def pad_sents(sents, pad_token, return_tensor = False):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []
    maxLen = 0
    ### YOUR CODE HERE (~6 Lines)
    for i in sents:
        maxLen = max(len(i),maxLen)
    for i in range(len(sents)):
        sen = sents[i].cpu().numpy().tolist()
        for j in range(maxLen - len(sen)):
            sen.append(pad_token)
        sen = torch.tensor(sen, dtype=torch.long).cuda()
        sents_padded.append(sen)
    if return_tensor:
        t = torch.zeros(len(sents), maxLen).long()
        for i in range(len(sents)):
            t[i] = sents_padded[i]
        sents_padded = t.cuda()

    return sents_padded


def pad_enc(x0, x1, sep):
    b = len(x0)
    maxLen = 0
    length = []
    for i in range(b):
        maxLen = max(x0[i].shape[0], maxLen)
        maxLen = max(x1[i].shape[0], maxLen)
    x = torch.zeros(b, int(2*maxLen+1), x0[0].shape[1])
    for i in range(b):
        h = torch.cat((x0[i], sep[i]), 0)
        h = torch.cat((h, x1[i]), 0)
        x[i, :h.shape[0]] = h
        length.append(h.shape[0])
    return x.cuda(), length
        

def pack_sequences(X, lengths, order=None):
    
    #X = [x.squeeze(0) for x in X]
    
    n = len(X)#2*batchsize
    #lengths = np.array([len(x) for x in X])
    if order is None:
        order = np.argsort(lengths)[::-1]#从后向前取反向的元素，直接在argsort里面调不就好了？本来是从小到大，现在是从大到小
    m = max(len(x) for x in X)
    
    X_block = X[0].new(n,m).zero_()
    
    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i,:len(x)] = x
        
    #X_block = torch.from_numpy(X_block) 
        
    lengths = lengths[order]#从大到小排序长度
    
    X = pack_padded_sequence(X_block, lengths, batch_first=True)#X_block要从大到小排好序，length也是，batch_first=True是说第一维是batch_size
    
    return X, order


def unpack_sequences(X, order):
    X,lengths = pad_packed_sequence(X, batch_first=True)
    X_block = torch.zeros(size=X.size())
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i]
    X_block = X_block.cuda()
    return X_block

def unpack_torchtensor(X, order):
    X_block = torch.zeros(size=X.size())
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i]
    X_block = X_block.cuda()
    return X_block


def collate_lists(args):
    x = [a[0] for a in args]
    y = [a[1] for a in args]
    return x, y


class ContactMapDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, augment=None, fragment=False, mi=64, ma=500):
        self.X = X
        self.Y = Y
        self.augment = augment
        self.fragment = fragment
        self.mi = mi
        self.ma = ma
        """
        if fragment: # multiply sequence occurence by expected number of fragments
            lengths = np.array([len(x) for x in X])
            mi = np.clip(lengths, None, mi)
            ma = np.clip(lengths, None, ma)
            weights = 2*lengths/(ma + mi)
            mul = np.ceil(weights).astype(int)
            X_ = []
            Y_ = []
            for i,n in enumerate(mul):
                X_ += [X[i]]*n
                Y_ += [Y[i]]*n
            self.X = X_
            self.Y = Y_
        """

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y[i]
        if self.fragment and len(x) > self.mi:
            mi = self.mi
            ma = min(self.ma, len(x))
            l = np.random.randint(mi, ma+1)
            i = np.random.randint(len(x)-l+1)
            xl = x[i:i+l]
            yl = y[i:i+l,i:i+l]
            # make sure there are unmasked observations
            while torch.sum(yl >= 0) == 0:
                l = np.random.randint(mi, ma+1)
                i = np.random.randint(len(x)-l+1)
                xl = x[i:i+l]
                yl = y[i:i+l,i:i+l]
            y = yl.contiguous()
            x = xl
        if self.augment is not None:
            x = self.augment(x)
        return x, y




class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, X0, X1, Y):
        self.X0 = X0
        self.X1 = X1
        self.Y = Y

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.Y[i]
    
class pre_Dataset(torch.utils.data.Dataset):
    def __init__(self, X0, X1):
        self.X0 = X0
        self.X1 = X1
    

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i]

def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)










