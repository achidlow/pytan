import sys

import numpy as np
from numpy.linalg import solve

from pytan._base import BayesNetClassifier
from pytan._utils import genvar, lognorm_pdf


class CLGBayesNetClassifier(BayesNetClassifier):
    def _preprocess(self, X):
        return np.atleast_3d(X).astype(np.float64)

    def _mutual_information(self, X_k):
        n_features = X_k.shape[1]
        mutual_info = np.zeros((n_features, n_features))
        for i in range(n_features):
            a = X_k[:, i]
            s_aa = genvar(a)
            for j in range(i + 1, n_features):
                b = X_k[:, j]
                s_ab = genvar(a, b)
                if s_ab == 0:
                    # perfectly correlated
                    mi = sys.float_info.max
                else:
                    mi = 0.5 * np.log(s_aa * genvar(b) / s_ab)
                mutual_info[i, j] = mi
        return mutual_info

    def _partial_fit(self, graph, X: np.ndarray):
        cpds = []
        for i, parent in enumerate(graph):
            if parent < 0:
                μ = X[:, i].mean(axis=0, keepdims=True)
                σ = X[:, i].std(axis=0, keepdims=True, ddof=1)
                cpds.append((μ, σ))
            else:
                params = fit_multivariate_linear_gaussian(X[:, i], X[:, parent])
                cpds.append(params)
        return cpds

    def _encode(self, X):
        return self._preprocess(X)

    def _partial_log_proba(self, graph, cpds, X: np.ndarray) -> np.ndarray:
        N, n_features, n_vars = X.shape
        ll = np.empty((n_features, N))
        for i, cpd in enumerate(cpds):
            parent = graph[i]
            if parent < 0:
                μ, σ = cpd
            else:
                β, β0, σ = cpd
                μ = X[:, parent] @ β + β0
            ll[i] = lognorm_pdf(X[:, i], μ, σ).sum(axis=1)
        return ll.sum(axis=0)


def fit_multivariate_linear_gaussian(Y, X):
    N, n_var = Y.shape

    μ_y = np.mean(Y, axis=0, keepdims=True)
    μ_x = np.mean(X, axis=0, keepdims=True)

    # compute covariances
    Yc = Y - μ_y
    Xc = X - μ_x
    Σ_xx = (Xc.T @ Xc) / (N - 1)  # type: np.ndarray
    Σ_xy = (Xc.T @ Yc) / (N - 1)  # type: np.ndarray

    β = solve(Σ_xx, Σ_xy)
    β0 = μ_y - μ_x @ β

    var_y = np.einsum('ij,ij->j', Yc, Yc) / (N - 1)  # ~2x faster than square and sum
    var = var_y - np.einsum('ij,ij->j', β, Σ_xy)
    σ = np.sqrt(var).reshape(μ_y.shape)

    return β, β0, σ
