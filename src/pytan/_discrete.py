from typing import List, Iterable, Optional
from math import log

import numpy as np

from pytan._base import BayesNetClassifier
from pytan._utils import FeatureIntegerizer


class DiscreteBayesNetClassifier(BayesNetClassifier):
    def __init__(self,
                 structure: str='tree',
                 conditional_fit: bool=False,
                 alpha: float=1.0,
                 values: List[Iterable[int]]=None,
                 graph: np.ndarray=None,
                 priors: Optional[np.ndarray]=None):
        super().__init__(structure=structure, conditional_fit=conditional_fit, graph=graph, priors=priors)
        self.alpha = alpha
        self.values = values

    def _preprocess(self, X: np.ndarray):
        self.preprocess_ = FeatureIntegerizer(values=self.values)
        return self.preprocess_.fit_transform(X)

    def _mutual_information(self, X_k):
        N, n_features = X_k.shape
        mutual_info = np.zeros((n_features, n_features))
        max_card = X_k.max() + 1
        column_counts = np.zeros((n_features, max_card))
        for col_idx in range(n_features):
            vals, counts = np.unique(X_k[:, col_idx], return_counts=True)
            column_counts[col_idx, vals] += counts
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi = 0
                ij_vals, ij_counts = np.unique(X_k[:, (i, j)], return_counts=True, axis=0)
                for ij_val, ijc in zip(ij_vals, ij_counts):
                    ic = column_counts[i, ij_val[0]]
                    jc = column_counts[j, ij_val[1]]
                    mi += ijc * log(N * ijc / (ic * jc))
                mutual_info[i, j] = mi
        return mutual_info

    def _partial_fit(self, graph, X):
        cpds = []
        card = self.preprocess_.card_
        for idx, n_vals in enumerate(card):
            parent = graph[idx]
            alpha = self.alpha
            if parent < 0:
                cpd = fit_multinomial(n_vals, X[:, idx], alpha)
            else:
                n_parent_vals = card[parent]
                alpha /= n_parent_vals
                X_parent_col = X[:, parent]
                cpd = np.vstack(fit_multinomial(n_vals, X[X_parent_col == p, idx], alpha)
                                for p in range(n_parent_vals))
            cpds.append(cpd)
        return cpds

    def _encode(self, X):
        return self.preprocess_.transform(X)

    def _partial_log_proba(self, graph, cpds, X: np.ndarray):
        N, n_features = X.shape
        l = np.empty((N, n_features))
        for idx, cpd in enumerate(cpds):
            parent = graph[idx]
            if parent < 0:
                l[:, idx] = cpd[X[:, idx]]
            else:
                l[:, idx] = cpd[X[:, parent], X[:, idx]]
        ll = np.log(l)
        return ll.sum(axis=1)


def fit_multinomial(n_vals, Xi, alpha):
    params = np.zeros(n_vals) + alpha
    vals, counts = np.unique(Xi, return_counts=True)
    params[vals] += counts
    params /= params.sum()
    return params
