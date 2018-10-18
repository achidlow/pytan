import abc
from typing import Optional, List, Any

import numpy as np
from sklearn import naive_bayes
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

from pytan._utils import maximum_spanning_directed_tree


class BayesNetClassifier(naive_bayes.BaseNB, metaclass=abc.ABCMeta):
    def __init__(self,
                 structure: str='tree',
                 conditional_fit: bool=False,
                 graph: np.ndarray=None,
                 priors: Optional[np.ndarray]=None):
        self.structure = structure
        self.conditional_fit = conditional_fit
        self.graph = graph
        self.priors = priors

    def _joint_log_likelihood(self, X):
        check_is_fitted(self, ('classes_', 'models_', 'priors_', 'n_features_', 'graph_'))
        X: np.ndarray = check_array(X, allow_nd=True)
        N, n_features = X.shape[:2]
        if n_features != self.n_features_:
            raise ValueError(f'Number of features {n_features} does not match previous data {self.n_features_}.')
        X = self._encode(X)
        ll = np.empty((N, self.n_classes_))
        log_priors = np.log(self.priors_)
        for k, (graph, model, log_prior) in enumerate(zip(self.graph_, self.models_, log_priors)):
            ll[:, k] = log_prior + self._partial_log_proba(graph, model, X)
        return ll

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, allow_nd=True)
        check_classification_targets(y)
        self.n_features_ = X.shape[1]
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.n_classes_ = len(self.classes_)
        self.priors_ = self.priors or (class_counts / np.sum(class_counts))

        X = self._preprocess(X)

        graph = self.graph
        if graph is None:
            if self.structure == 'naive':
                graph = self._naive_structure()
            elif self.structure == 'tree':
                graph = self._chow_liu_structure(X, y)
            else:
                raise ValueError(f'Invalid structure type: {self.structure}')
        self.graph_ = graph

        self.models_ = [
            self._partial_fit(graph_k, X[y == cls])
            for graph_k, cls in zip(graph, self.classes_)
        ]
        return self

    def _naive_structure(self):
        return -np.ones((self.n_classes_, self.n_features_), dtype=np.int8)

    def _chow_liu_structure(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_features = self.n_features_
        class_data = (X[y == cls] for cls in self.classes_)
        graph = self._naive_structure()
        if self.conditional_fit:
            for k, X_k in enumerate(class_data):
                weighted_edges = self._mutual_information(X_k)
                graph[k] = maximum_spanning_directed_tree(n_features, weighted_edges)
        else:
            weighted_edges = np.zeros((n_features, n_features))
            for X_k in class_data:
                weighted_edges += self._mutual_information(X_k)
            tree = maximum_spanning_directed_tree(n_features, weighted_edges)
            graph[:] = tree.reshape((1, n_features))
        return graph

    @abc.abstractmethod
    def _preprocess(self, X):
        raise NotImplementedError()

    @abc.abstractmethod
    def _mutual_information(self, X_k):
        raise NotImplementedError()

    @abc.abstractmethod
    def _partial_fit(self, graph, X) -> List[Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _encode(self, X):
        raise NotImplementedError()

    @abc.abstractmethod
    def _partial_log_proba(self, graph, cpds, X) -> np.ndarray:
        raise NotImplementedError()
