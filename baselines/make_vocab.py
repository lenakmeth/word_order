import os
import json

filepath = '/users/melodi/emetheni/word_order/data/frwac_jan/'

train = filepath + 'train.tsv'
words = []

with open(train, 'r') as f:
    for line in f:
        words += line.split('\t')[0].lower().split(' ')
        
vocab = {word:n for n, word in enumerate(set(words), 2)}
vocab['<unk>'] = 1
vocab['<pad>'] = 0
with open('vocab.json', 'w') as f:
    json.dump(vocab, f, ensure_ascii=False)
