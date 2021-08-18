"""
Created on Wed Aug 19 10:44:56 2020

@author: tushar
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


data_path = '/Users/tushar/Documents/CS677 project/final data/5yeardata.csv'
data = pd.read_csv(data_path)
data.dropna(inplace=True)

features = data.loc[:, (data.columns!= 'Player1')&(data.columns!= 'Player2')&(data.columns!= 'player1_name')&(data.columns!= 'player2_name')&(data.columns!= 'winner')]

X = features[features.columns].values

Y = data[['winner']].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3, random_state=3)

model = LogisticRegression(max_iter=4000)
model.fit(X_train,Y_train)



pickle.dump(model, open('model.pkl','wb'))
