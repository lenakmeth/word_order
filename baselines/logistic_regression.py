#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:57:53 2021

@author: lena
"""

from utils import *
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

max_len = 128
data_filepath = '/users/melodi/emetheni/word_order/data/frwac_jan/'

X_train, y_train = make_sets(data_filepath + 'train.tsv', max_len)
X_val, y_val = make_sets(data_filepath + 'dev.tsv',  max_len)
X_test, y_test = make_sets(data_filepath + 'test.tsv', max_len)

# Logistic regression baseline with scaled data
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)

# score = pipe.score(X_val, y_val)
# print(score)

val_preds = pipe.predict(X_val)
test_preds = pipe.predict(X_test)

print('log-reg\n')
# classification report 
print('\nValidation:')
print(classification_report(y_val, val_preds))

print('Test:')
print(classification_report(y_test, test_preds))