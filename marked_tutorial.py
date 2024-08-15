# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:28:28 2024

@author: ajult
"""

import pandas as pd
import numpy as np

columns = [f'V{i}' for i in range(1, 29)]

# add more col name to columns
columns.insert(0, 'time')
columns.append('amount')
columns.append('target')
columns
# read data
df = pd.read_csv('./data/card.csv', names=columns)



# get the predictor and traget variables

X = df.loc[:, 'time':'amount']
y = df['target']

# shape pf the data
X.shape, y.shape

# test data into train_validate and test
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(1/8), random_state=33)


# shape of the data
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

# import the required MLPclassifier library
from sklearn.neural_network import MLPClassifier

# create the model
mlp_classifier = MLPClassifier(verbose=True).fit(X_train, y_train)
print("Trainig score: ", mlp_classifier.score(X_train, y_train))

# validation score
print("Validation score: ", mlp_classifier.score(X_val, y_val))

# test score
print("Test score: ", mlp_classifier.score(X_test, y_test))

mlp_classifier.get_params()

