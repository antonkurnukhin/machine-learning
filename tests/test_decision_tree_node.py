import unittest

from src.decision_tree.node import Node


class TestNode(unittest.TestCase):

    def test_node_initialization(self):
        # Тестирование инициализации с заданными параметрами
        node = Node(feature_index=1, threshold=0.5, value=10)
        self.assertEqual(node.feature_index, 1)
        self.assertEqual(node.threshold, 0.5)
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)
        self.assertEqual(node.value, 10)

        # Тестирование инициализации по умолчанию
        default_node = Node()
        self.assertIsNone(default_node.feature_index)
        self.assertIsNone(default_node.threshold)
        self.assertIsNone(default_node.left)
        self.assertIsNone(default_node.right)
        self.assertIsNone(default_node.value)

    def test_is_leaf_node(self):
        # Тестирование листового узла
        leaf_node = Node(value=10)
        self.assertTrue(leaf_node.is_leaf_node())

        # Тестирование не листового узла
        non_leaf_node = Node(feature_index=1, threshold=0.5)
        self.assertFalse(non_leaf_node.is_leaf_node())


if __name__ == '__main__':
    unittest.main()
