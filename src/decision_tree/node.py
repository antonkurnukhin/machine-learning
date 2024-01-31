class Node:
    def __init__(
            self,
            feature_index: int = None,
            threshold: float = None,
            left: "Node" = None,
            right: "Node" = None,
            value: float = None,
    ) -> None:
        """
        Конструктор класса Node.

        :param feature_index: Индекс признака, используемого для разделения данных в узле.
        :param threshold: Пороговое значение для разделения данных.
        :param left: Левый дочерний узел.
        :param right: Правый дочерний узел.
        :param value: Значение в листовом узле.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        """
        Проверка, является ли узел листовым.
        """
        return self.value is not None

