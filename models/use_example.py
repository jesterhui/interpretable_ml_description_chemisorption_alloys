"""
Example script that shows how to unpickle iGAM models and use them to make a
prediction.
"""
import pickle
import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import mean_squared_error


# read in data for adsorbate XX
DATA = np.loadtxt('../data/processed/XX_data.csv', dtype='float',
                  delimiter=',', skiprows=1, usecols=range(1, 11))
Y = DATA[:, 0].reshape(-1, 1)
X = DATA[:, [1, 5, 6, 7, 8, 9]]

# unpickle XX GAM
GAM = pickle.load(open('XX_gam.pickle', 'rb'))

# use gam to make prediction
Y_PRED = GAM.predict(X)

# print RMSE
print('RMSE = {}'.format(mean_squared_error(Y_PRED, Y)**.5))
