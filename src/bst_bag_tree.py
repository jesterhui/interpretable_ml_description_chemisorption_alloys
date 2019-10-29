"""
Intelligbible models for regression, from Lou, Caruana and Gehrke
(doi: 10.1145/2339530.2339556).

"""
import itertools as it
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class BoostedBaggedTreeGAM:
    """iGAM for inteliigible models.

    Attributes:
        model (dict): Dictionary containing constituent trees of model.
        m_boost (int): Number of iterations for gradient boosting.
        n_leaves (int): Number of leaves in each tree.
        n_trees (int): Number of bagged trees for each shaping function.

    """

    def __init__(self, m_boost, n_leaves=2, n_trees=100, pairwise=0):
        """

        Args:
            m_boost (int): Number of iterations for gradient boosting.
            n_leaves (int): Number of leaves in each tree.
            n_trees (int): Number of bagged trees for each shaping function.
            pairwise(int): Numer of pairwise interactions to include.

        """
        self.model = {}
        self.pairwise = pairwise
        self.pairwise_model = {}
        self.pairwise_inds = {}
        self.m_boost = m_boost
        self.n_leaves = n_leaves
        self.n_trees = n_trees
        self.y_avg = None

    def fit(self, x_train, y_train):
        """Train model.

        Args:
            x_train (obj): (n, d) NumPy array containing training input
            samples.
            y_train (obj): (n, 1) Numpy vector containing target values.

        """
        self.y_avg = np.mean(y_train)
        n_samples, d_features = x_train.shape
        # build model dictionary
        for j in range(d_features):
            self.model[j] = []

        # fill model dictionary with decision trees
        for _ in range(self.m_boost):
            for j in range(d_features):
                bagged_trees = []
                for _ in range(self.n_trees):
                    # bootstrap sampling
                    ind = np.random.randint(n_samples,
                                            size=n_samples)
                    x_sample = x_train[ind, :]
                    # boosting
                    y_sample = (y_train[ind, :].reshape(-1, 1)
                                - self.predict(x_sample))
                    x_sample = x_train[ind, j].reshape(-1, 1)
                    f_j = DecisionTreeRegressor(max_leaf_nodes=self.n_leaves)
                    f_j.fit(x_sample, y_sample)
                    bagged_trees.append(f_j)
                self.model[j].append(bagged_trees)

        if self.pairwise > 0:
            self.train_pairwise(x_train, y_train, self.pairwise)

    def predict(self, x_pred):
        """Use learned model to label data.

        Args:
            x_pred (obj): (n, d) NumPy array of test samples.

        Returns:
            obj: (n, 1) NumPy vector containging predictions.

        """
        # sum over model contributions for each feature
        y_pred = self.y_avg
        for j in self.model:
            for m_iter in self.model[j]:
                for f_j in m_iter:
                    y_pred = y_pred + (f_j.predict(x_pred[:, j].reshape(-1, 1))
                                       .reshape(-1, 1) / self.n_trees)
        for key in self.pairwise_inds:
            x_pred_p = x_pred[:, self.pairwise_inds[key]]
            for f_j in self.pairwise_model[key]:
                y_pred = y_pred + (f_j.predict(x_pred_p)
                                   .reshape(-1, 1) / self.n_trees)

        return y_pred

    def feature_contribution(self, x_feat, j_feat):
        """Obtain contribution of individual feature to overall prediction.

        Args:
            x_feat (type): Feature data.
            j_feat (int): Index of feature.

        Returns:
            obj: (n, 1) NumPy vector containing feature contrbution.

        """
        # peform prediction for single variable
        y_feat = 0
        for m_iter in self.model[j_feat]:
            for f_j in m_iter:
                y_feat = y_feat + (f_j.predict(x_feat[:, j_feat].reshape(-1, 1)
                                               ).reshape(-1, 1) / self.n_trees)
        return y_feat

    def pair_contribution(self, x_feat, j_pair):
        """Obtain contribution of individual feature to overall prediction.

        Args:
            x_feat (type): Feature data.
            j_feat (int): Index of feature.

        Returns:
            obj: (n, 1) NumPy vector containing feature contrbution.

        """
        # peform prediction for single variable
        y_pair = 0
        print(self.pairwise_inds[j_pair])
        x_pred_p = x_feat[:, self.pairwise_inds[j_pair]]
        for f_j in self.pairwise_model[j_pair]:
            y_pair = y_pair + (f_j.predict(x_pred_p)
                               .reshape(-1, 1) / self.n_trees)
        return y_pair

    def get_weights(self, x_w):
        """Get feature weights of iGAM model.

        Args:
            x_w (obj): NumPy array containing samples used to estimate weights.

        Returns:
            obj: (n, ) NumPy vector containing weights.

        """
        n_samples, n_features = x_w.shape
        weights = np.zeros(n_features,)
        for j in range(n_features):
            f_x = self.feature_contribution(x_w, j)
            weights[j] = np.sqrt((1 / n_samples) * np.sum(f_x ** 2))
        return weights

    def train_pairwise(self, x_train, y_train, n_pairs):
        """Train pairwise interactions.

        Args:
            x_train (obj): (n, d) NumPy array containing training input
            samples.
            y_train (obj): (n, 1) Numpy vector containing target values.
            n_pairs (int): Numer of pairwise interactions to include.

        """
        n_samples, d_features = x_train.shape
        possible_pairs = list(it.combinations(range(d_features), 2))
        weights = self.get_weights(x_train)
        sorted_weights = np.argsort(np.flip(np.argsort(weights)))
        ranking = []
        for pair in possible_pairs:
            ranking.append(sorted_weights[pair[0]] + sorted_weights[pair[1]])
        ranking = np.asarray(ranking)
        for i in range(n_pairs):
            best = np.argsort(ranking)[i]
            best = possible_pairs[best]
            bagged_trees = []
            for _ in range(self.n_trees):
                # bootstrap sampling
                ind = np.random.randint(n_samples,
                                        size=n_samples)
                x_sample = x_train[ind, :]
                # boosting
                y_sample = (y_train[ind, :].reshape(-1, 1)
                            - self.predict(x_sample))
                x_sample = x_sample[:, best]
                f_j = DecisionTreeRegressor(max_leaf_nodes=4)
                f_j.fit(x_sample, y_sample)
                bagged_trees.append(f_j)

            self.pairwise_model[i] = bagged_trees
            self.pairwise_inds[i] = best
