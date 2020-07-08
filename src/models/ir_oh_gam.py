"""
Generate OH model for Ir surface.
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import setup
from src.bst_bag_tree import BoostedBaggedTreeGAM

np.random.seed()
DATA = np.loadtxt('../../data/processed/ir_oh_data.csv', dtype='float',
                  delimiter=',', skiprows=1, usecols=range(1, 11))
Y = DATA[:, 0].reshape(-1, 1)
X = DATA[:, [1, 5, 6, 7, 8, 9]]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.3,
                                                    shuffle=True)
GAM = BoostedBaggedTreeGAM(m_boost=2, n_leaves=2, n_trees=100, pairwise=3)
GAM.fit(X_TRAIN, Y_TRAIN)
pickle.dump(GAM, open('../../models/ir_oh_gam.pickle', 'wb'))
