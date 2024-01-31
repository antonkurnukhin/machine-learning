from abc import ABCMeta, abstractmethod

import numpy as np

from .node import Node


class DecisionTree(metaclass=ABCMeta):
    def __init__(self):
        self.max_depth = None
        self.root = None

    def fit(self, x_train, y_train):
        self.root = self.__build_tree(x_train, y_train)

    def predict(self, x_test):
        return [self.__predict(self.root, x) for x in x_test]

    def __build_tree(self, x, y, depth=0):
        n_samples, n_features = len(y), len(x[0])

        # Критерии остановки
        if depth >= self.max_depth or n_samples <= 2 or len(np.unique(y)) == 1:
            leaf_value = self.__calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Нахождение наилучшего разделения
        best_feat, best_thresh = self.__best_split(x, y, n_features)

        if best_feat is None:
            leaf_value = self.__calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Разделение датасета
        left_idxs, right_idxs = self.__split(x[:, best_feat], best_thresh)
        left_subtree = self.__build_tree(x[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self.__build_tree(x[right_idxs, :], y[right_idxs], depth + 1)

        return Node(
            feature_index=best_feat,
            threshold=best_thresh,
            left=left_subtree,
            right=right_subtree,
        )

    def __predict(self, node, x):
        # Если достигнут листовой узел
        if node.is_leaf_node():
            return node.value

        # Рекурсивный спуск в левое или правое поддерево
        if x[node.feature_index] < node.threshold:
            return self.__predict(node.left, x)
        return self.__predict(node.right, x)

    def __best_split(self, x, y, n_features):
        best_metric_value = float("inf")
        best_feat, best_thresh = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(x[:, feature_index])
            for threshold in thresholds:
                metric = self.__calculate_metric(x, y, feature_index, threshold)
                if metric < best_metric_value:
                    best_metric_value = metric
                    best_feat = feature_index
                    best_thresh = threshold

        return best_feat, best_thresh

    @staticmethod
    def __split(feature_values, threshold):
        left_ids = np.where(feature_values < threshold)[0]
        right_ids = np.where(feature_values >= threshold)[0]
        return left_ids, right_ids

    @staticmethod
    @abstractmethod
    def __calculate_leaf_value(y):
        raise NotImplementedError

    @abstractmethod
    def __calculate_metric(self, x, y, feature_index, threshold):
        raise NotImplementedError
