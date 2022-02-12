#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:30:22 2021

@author: lena
"""

import numpy as np
import json
from tqdm import tqdm_notebook
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1 = UNK
# 0 = PAD


def encode_sequence(sent, vocab):
    """encode sequence with our vocab"""
    
    encoded = []
    for word in sent.split(' '):
        if word in vocab:
            encoded.append(vocab[word])
        else:
            encoded.append(1) #UNK
    return encoded


def make_sets(type_set, max_seq):
    
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
   
    X = []
    y = []

    with open(type_set, 'r') as f:
        for line in f:
            encoded_seq = encode_sequence(line.split('\t')[0], vocab)
            encoded_seq = encoded_seq + [0]*(128-len(encoded_seq))
            
            assert len(encoded_seq) == max_seq
            
            X.append(encoded_seq)
            if line.strip().split('\t')[-1] == 'before':
                y.append(0)
            else:
                y.append(1)

    return np.asarray(X), np.asarray(y)


def load_pretrained_vectors(word2idx, fname):
    """Load pretrained vectors and create embedding layers.
    
    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

    print("Loading pretrained vectors...")
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    for line in tqdm_notebook(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings


def data_loader(inputs, labels, batch_size=50):
    """Convert train and validation sets to torch.Tensors and load them to
    DataLoader.
    """
    
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)

    # Specify batch_size
    batch_size = 50

    # Create DataLoader for training data
    data = TensorDataset(inputs, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def predicted_label(encoded_sent, model, max_len=62):
    """Predict probability that a review is positive."""

        # Convert to PyTorch tensors
    input_id = torch.tensor(encoded_sent).unsqueeze(dim=0)

    # Compute logits
    logits = model.forward(input_id)

    #  Compute probability
    probs = F.softmax(logits, dim=1).squeeze(dim=0)
    
    if probs[0] > 0.5:
        return 0
    else:
        return 1
    