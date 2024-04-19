import pandas as pd
import numpy as np
import matplotlib as plt

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values()