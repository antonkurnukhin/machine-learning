from abc import ABC
from typing import Optional

import numpy as np

from .base import DecisionTree
from .node import Node


class DecisionTreeRegressor(DecisionTree, ABC):
    def __init__(self, max_depth: int = 10):
        super(DecisionTree, self).__init__()
        self.max_depth = max_depth
        self.root: Optional[Node] = None

    @staticmethod
    def _DecisionTree__calculate_leaf_value(y):
        return np.mean(y)

    @staticmethod
    def _DecisionTree__calculate_metric(X, y, feature_index, threshold):
        left_ids = X[:, feature_index] < threshold
        right_ids = X[:, feature_index] >= threshold

        # Если одна из групп пуста, возвращаем "бесконечность" для MSE
        if len(y[left_ids]) == 0 or len(y[right_ids]) == 0:
            return float("inf")

        metric_left = np.mean((y[left_ids] - np.mean(y[left_ids])) ** 2)
        metric_right = np.mean((y[right_ids] - np.mean(y[right_ids])) ** 2)

        # Взвешенное среднее MSE для двух групп
        n_left, n_right = len(y[left_ids]), len(y[right_ids])
        n_total = len(y)
        weighted_metric = (n_left / n_total) * metric_left + (n_right / n_total) * metric_right

        return weighted_metric


class DecisionTreeClassifier(DecisionTree, ABC):
    def __init__(self, max_depth: int = 10):
        super(DecisionTree, self).__init__()
        self.max_depth = max_depth
        self.root: Optional[Node] = None

    @staticmethod
    def _DecisionTree__calculate_leaf_value(y):
        return np.bincount(y).argmax()

    def _DecisionTree__calculate_metric(self, x, y, feature_index, threshold) -> float:
        left_ids = x[:, feature_index] < threshold
        right_ids = x[:, feature_index] >= threshold

        # Если одна из групп пуста, возвращаем "бесконечность" для индекса Джини
        if len(y[left_ids]) == 0 or len(y[right_ids]) == 0:
            return float("inf")

        metric_left = self._calculate_gini_index(y[left_ids])
        metric_right = self._calculate_gini_index(y[right_ids])

        # Взвешенное среднее значение Джини для двух групп
        n_left, n_right = len(y[left_ids]), len(y[right_ids])
        n_total = len(y)
        weighted_metric = (n_left / n_total) * metric_left + (
                n_right / n_total
        ) * metric_right

        return weighted_metric

    @staticmethod
    def _calculate_gini_index(y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - sum(probabilities ** 2)
        return gini
