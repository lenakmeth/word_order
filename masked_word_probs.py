#!/usr/bin/env python
# coding: utf-8

from transformers import CamembertModel, CamembertTokenizer, pipeline
import spacy
import json
import matplotlib.pyplot as plt
import sys

modelname = sys.argv[1]

tok = CamembertTokenizer.from_pretrained(modelname)  
nlp = spacy.load('fr_core_news_sm')
camembert_fill_mask = pipeline("fill-mask", model=modelname, tokenizer=modelname, top_k = 50)

def open_file(file):
    sentences = []
    labels = []

    with open(file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            l = line.strip().split('\t')
            if select_sentences(l, tok):
                if l[-1] == 'before':
                    labels.append(0)
                else:
                    labels.append(1)
                sentences.append(select_sentences(l, tok))

    return sentences, labels

def select_sentences(sentence, tokenizer):
        
    tok_noun = tokenizer.encode(sentence[2])
    tok_adj = tokenizer.encode(sentence[1])
            
    if len(tok_noun) == 3 and len(tok_adj) == 3:
        sent = sentence[0].split(' ')
        idx = sent.index(sentence[1])
        sent[idx] = '<mask>'
        final_sent = [' '.join(sent)] + sentence[1:]
        return final_sent
    else:
        return None

def pos_tagger(result):
    # one word
    doc = nlp(result['sequence'])
    adj = result['token_str'].replace('▁', '')
    for token in doc:
        if token.text == adj:
            if token.pos_ == None:
                print(result)
            return token.pos_

sentences, labels = open_file('data/frwac/test.tsv')
for sentence in sentences:
    add_results = []
    results = camembert_fill_mask(sentence[0])
    for result in results:
        add_results.append([result['token_str'].replace('▁', ''), 
                            result['score'], pos_tagger(result)])
    sentence.append(add_results)


with open('mask_adj_camembert-large.json', 'w') as f:
    json.dump(sentences, f, ensure_ascii= False)
