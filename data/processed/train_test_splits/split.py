"""
Generate O model for Ag surface.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)
DATA = np.loadtxt('../rh_o_data.csv', dtype='float',
                  delimiter=',', skiprows=1, usecols=range(1, 11))
Y = np.arange(DATA.shape[0])
X = DATA[:, [1, 5, 6, 7, 8, 9]]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.3,
                                                    shuffle=True)

SPLITS = np.zeros(DATA.shape[0])
SPLITS[Y_TRAIN] = 0
SPLITS[Y_TEST] = 1
index = pd.read_csv('../rh_o_data.csv')
split_df = pd.DataFrame(index=index['index'])
split_df['split'] = SPLITS
split_df.to_csv('rh_train_test_split.csv')

np.random.seed(0)
DATA = np.loadtxt('../pd_o_data.csv', dtype='float',
                  delimiter=',', skiprows=1, usecols=range(1, 11))
Y = np.arange(DATA.shape[0])
X = DATA[:, [1, 5, 6, 7, 8, 9]]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.3,
                                                    shuffle=True)

SPLITS = np.zeros(DATA.shape[0])
SPLITS[Y_TRAIN] = 0
SPLITS[Y_TEST] = 1
index = pd.read_csv('../pd_o_data.csv')
split_df = pd.DataFrame(index=index['index'])
split_df['split'] = SPLITS
split_df.to_csv('pd_train_test_split.csv')

np.random.seed(0)
DATA = np.loadtxt('../ag_o_data.csv', dtype='float',
                  delimiter=',', skiprows=1, usecols=range(1, 11))
Y = np.arange(DATA.shape[0])
X = DATA[:, [1, 5, 6, 7, 8, 9]]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.3,
                                                    shuffle=True)

SPLITS = np.zeros(DATA.shape[0])
SPLITS[Y_TRAIN] = 0
SPLITS[Y_TEST] = 1
index = pd.read_csv('../ag_o_data.csv')
split_df = pd.DataFrame(index=index['index'])
split_df['split'] = SPLITS
split_df.to_csv('ag_train_test_split.csv')

np.random.seed(0)
DATA = np.loadtxt('../ir_o_data.csv', dtype='float',
                  delimiter=',', skiprows=1, usecols=range(1, 11))
Y = np.arange(DATA.shape[0])
X = DATA[:, [1, 5, 6, 7, 8, 9]]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.3,
                                                    shuffle=True)

SPLITS = np.zeros(DATA.shape[0])
SPLITS[Y_TRAIN] = 0
SPLITS[Y_TEST] = 1
index = pd.read_csv('../ir_o_data.csv')
split_df = pd.DataFrame(index=index['index'])
split_df['split'] = SPLITS
split_df.to_csv('ir_train_test_split.csv')

np.random.seed(0)
DATA = np.loadtxt('../pt_o_data.csv', dtype='float',
                  delimiter=',', skiprows=1, usecols=range(1, 11))
Y = np.arange(DATA.shape[0])
X = DATA[:, [1, 5, 6, 7, 8, 9]]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.3,
                                                    shuffle=True)

SPLITS = np.zeros(DATA.shape[0])
SPLITS[Y_TRAIN] = 0
SPLITS[Y_TEST] = 1
index = pd.read_csv('../pt_o_data.csv')
split_df = pd.DataFrame(index=index['index'])
split_df['split'] = SPLITS
split_df.to_csv('pt_train_test_split.csv')

np.random.seed(0)
DATA = np.loadtxt('../au_o_data.csv', dtype='float',
                  delimiter=',', skiprows=1, usecols=range(1, 11))
Y = np.arange(DATA.shape[0])
X = DATA[:, [1, 5, 6, 7, 8, 9]]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.3,
                                                    shuffle=True)

SPLITS = np.zeros(DATA.shape[0])
SPLITS[Y_TRAIN] = 0
SPLITS[Y_TEST] = 1
index = pd.read_csv('../au_o_data.csv')
split_df = pd.DataFrame(index=index['index'])
split_df['split'] = SPLITS
split_df.to_csv('au_train_test_split.csv')
