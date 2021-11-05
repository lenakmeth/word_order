import io
import json
import os
from transformers import CamembertTokenizer, FlaubertTokenizer
from configure import parse_args
import time
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re

args = parse_args()


def read_parser_file(file):
    """ Opens the file with the parser output."""

    # read and format sentences
    sents = []
    lines = []

    marker = False
    with open(file, "r") as f:
        for line in f:
            line = re.sub("\s{2,}", "\t", line)
            lines.append(line)

    for line in lines:
        if line.startswith("#"):
            pass
        elif line.split("\t")[0].isdigit():
            marker = True
            if line.split("\t")[0] == "1":  # first word of new sentence
                sents.append({})
            if marker:
                try:
                    sents[-1].update(format_line(line))
                except IndexError:
                    pass
        else:
            marker = False

    return sents


def format_line(raw_line):
    """ Formats word features into a list of features. Format:
        {tokenID: { word:X, POS:Y, label:x, headID:0}, ...} """

    raw_features = raw_line.strip().split("\t")
    features = {
        "word": None,
        "lemma": None,
        "POS": None,
        "TIGER": None,
        "headID": None,
        "label": None,
    }

    tokenID = raw_features[0]
    features["word"] = raw_features[1].replace("_", "").lower()
    features["lemma"] = raw_features[2].replace("_", "").lower()
    features["POS"] = raw_features[3].replace("_", "")
    features["TIGER"] = raw_features[4].replace("_", "")
    if raw_features[6].isdigit():
        features["headID"] = int(raw_features[6])
    else:
        try:
            features["headID"] = int(re.search("\d+", raw_features[6]).group(0).strip())
            features["label"] = (
                re.search("\D+", raw_features[6].replace("_", "")).group(0).strip()
            )
        except AttributeError:
#             print(raw_features)
            pass
    if not features["label"]:
        features["label"] = raw_features[7]

    output = {int(tokenID): features}

    return output


def read_sents(path):
    """ Read the .tsv files with the annotated sentences. 
        File format: sent_id, sentence, verb, verb_idx, label"""

    def open_file(file):
        sentences = []
        labels = []
        
        with open(file, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                l = line.strip().split('\t')
                sentences.append(l)
                if l[-1] == 'before':
                    labels.append(0)
                else: # after
                    labels.append(1)
                
            return sentences,labels
        
    train_sentences, train_labels = open_file(path + '/train.tsv')    
    val_sentences, val_labels = open_file(path + '/val.tsv')
    test_sentences, test_labels = open_file(path + '/test.tsv')

    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels

def read_sents_rg(path):
    """ Read the .tsv files with the annotated sentences. 
        File format: sent_id, sentence, verb, verb_idx, label"""

    def open_file(file):
        sentences = []
        labels = []
        
        with open(file, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                l = line.strip().split('\t')
                if l[-1] == 'before':
                    labels.append(0)
                else:
                    labels.append(1)
                sentences.append([l[0]+' </s> '+l[1]] + l[2:])

            return sentences, labels
        
    train_sentences, train_labels = open_file(path + '/train.tsv')    
    val_sentences, val_labels = open_file(path + '/val.tsv')
    test_sentences, test_labels = open_file(path + '/test.tsv')
#     test_sentences, test_labels = open_file(path + '/test_balanced.tsv')

    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels
            

def tokenize_and_pad(sentences):
    """ We are using .encode_plus. This does not make specialized attn masks 
        like in our selectional preferences experiment. Revert to .encode if
        necessary."""
    
    input_ids = []
    segment_ids = [] # token type ids
    attention_masks = []
    labels = []
    
    if 'flaubert' in args.transformer_model:
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tok = FlaubertTokenizer.from_pretrained(args.transformer_model )
    elif 'camembert' in args.transformer_model:
        tok = CamembertTokenizer.from_pretrained(args.transformer_model)
        
    for sentence_list in sentences:
        try:
            encoded_dict = tok.encode_plus(
                                sentence_list[0], # the sentence                  
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 128,      # Pad & truncate all sentences.
                                padding = 'max_length',
                                truncation = True,
                                return_attention_mask = True, # Construct attn. masks.
                                # return_tensors = 'pt',     # Return pytorch tensors.
                           )
            
            # Add segment ids, add 1 for verb idx
    #         segment_idx = encoded_dict['input_ids'][1:].index(0) + 1
            # this will only work for flaubert
#             segment_idx = sentence_list[0].split(' ').index('</s>') +1
#             segment_id = [0] * (segment_idx) + [1] * (len(encoded_dict['input_ids'])-segment_idx)
#             assert len(segment_id) == len(encoded_dict['input_ids'])
        
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
#             segment_ids.append(segment_id)
            segment_ids.append([0]*len(encoded_dict['input_ids']))
            
            if sentence_list[-1] == 'before':
                labels.append(0)
            elif sentence_list[-1] == 'after':
                labels.append(1)
            
        except AssertionError:
            print(sentence_list)
            pass

    return input_ids, attention_masks, segment_ids, labels


def tokenize_and_pad_with_attn(sentences):
    """ We are using .encode_plus. This does not make specialized attn masks 
        like in our selectional preferences experiment. Revert to .encode if
        necessary."""
    
    input_ids = []
    segment_ids = [] # token type ids
    attention_masks = []
    labels = []
    final_sentences = []
    
    if 'flaubert' in args.transformer_model:
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tok = FlaubertTokenizer.from_pretrained(args.transformer_model )
    elif 'camembert' in args.transformer_model:
        tok = CamembertTokenizer.from_pretrained(args.transformer_model)
    
    for sentence_list in sentences:
        try:
            noun = sentence_list[2]
            adj = sentence_list[1]
            
            tok_noun = tok.encode(noun)
            tok_adj = tok.encode(adj)
            
            if len(tok_noun) == 3 and len(tok_adj) == 3:
                encoded_dict = tok.encode_plus(
                                    sentence_list[0], # the sentence                  
                                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                    max_length = 128,      # Pad & truncate all sentences.
                                    padding = 'max_length',
                                    truncation = True,
                                    return_attention_mask = True, # Construct attn. masks.
                                    # return_tensors = 'pt',     # Return pytorch tensors.
                               )
                if tok_noun[1] in encoded_dict['input_ids'] and \
                    tok_adj[1] in encoded_dict['input_ids']:
                    
                    attn_mask = [0] * len(encoded_dict['input_ids'])
                    attn_mask[encoded_dict['input_ids'].index(tok_noun[1])] = 1
                    attn_mask[encoded_dict['input_ids'].index(tok_adj[1])] = 1
                    assert len(encoded_dict['input_ids']) == len(attn_mask)
                    
                    input_ids.append(encoded_dict['input_ids'])
                    segment_ids.append([0] * len(encoded_dict['input_ids']))
                    attention_masks.append(attn_mask)
                    final_sentences.append(sentence_list[:3])

                    if sentence_list[-1] == 'before':
                        labels.append(0)
                    elif sentence_list[-1] == 'after':
                        labels.append(1)
            
        except AssertionError:
            pass

    return input_ids, attention_masks, segment_ids, labels, final_sentences


def flat_accuracy(labels, preds):
    """ Function to calculate the accuracy of our predictions vs labels. """
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

    
def decode_result(encoded_sequence):

    if 'camembert' in args.transformer_model:
        tok = CamembertTokenizer.from_pretrained(args.transformer_model )
    elif 'flaubert' in args.transformer_model:
        tok = FlaubertTokenizer.from_pretrained(args.transformer_model )
    
    # decode + remove special tokens
    tokens_to_remove = ['[PAD]', '[SEP]', '[CLS]', '<pad>', '<sep>', '<s>', '</s>']
    decoded_sequence = [w.replace('Ġ', '').replace('▁', '').replace('</w>', '')
                        for w in list(tok.convert_ids_to_tokens(encoded_sequence))
                        if not w.strip() in tokens_to_remove]
    
    return ' '.join(decoded_sequence)
