from typing import Set, TypeVar

import numpy as np
from numpy.linalg import det
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array

__all__ = [
    'discard',
    'maximum_spanning_directed_tree',
    'lognorm_pdf',
    'genvar',
    'FeatureIntegerizer',
]

T = TypeVar('T')


def discard(s: Set[T], value: T) -> bool:
    count = len(s)
    s.discard(value)
    return len(s) != count


def maximum_spanning_directed_tree(num_nodes: int, weighted_edges: np.ndarray) -> np.ndarray:
    adjacencies = sparse.csr_matrix(-weighted_edges)
    mst = sparse.csgraph.minimum_spanning_tree(adjacencies)
    edges = {frozenset((i, j)) for i, j in zip(*mst.nonzero())}
    parent = -np.ones(num_nodes, dtype=np.int8)
    visited = {0}
    while edges:
        for i in range(1, num_nodes):
            for j in range(num_nodes):
                if j in visited and discard(edges, frozenset((i, j))):
                    visited.add(i)
                    parent[i] = j
    return parent


LN_SQRT_2PI = np.log(np.sqrt(2*np.pi))


def lognorm_pdf(x, mu, sigma):
    return -(x - mu)**2 / (2*sigma**2) - (LN_SQRT_2PI + np.log(sigma))


def genvar(a: np.ndarray, b: np.ndarray=None):
    """Generalised variance"""
    if b is None:
        X = a.copy()
    else:
        X = np.hstack((a, b))
    X -= np.mean(X, axis=0, keepdims=True)
    c = (X.T @ X) / (X.shape[0] - 1)  # type: np.ndarray
    if c.size == 1:
        return c.item()
    else:
        return det(c)


class FeatureIntegerizer(BaseEstimator, TransformerMixin):
    def __init__(self, values=None):
        self.values = values

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def _apply(self, X, func):
        X: np.ndarray = check_array(X)
        output = np.empty_like(X, dtype=np.int8)
        for idx, enc in enumerate(self.encoders_):
            output[:, idx] = func(enc, X[:, idx])
        return output

    @property
    def card_(self):
        return np.array([enc.classes_.size for enc in self.encoders_])

    def transform(self, X):
        return self._apply(X, LabelEncoder.transform)

    def fit_transform(self, X, y=None, **fit_params):
        self.encoders_ = [LabelEncoder() for _ in range(X.shape[1])]
        if self.values is None:
            return self._apply(X, LabelEncoder.fit_transform)
        else:
            for vals, enc in zip(self.values, self.encoders_):
                enc.classes_ = np.asarray(vals)
            return self.transform(X)
