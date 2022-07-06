"""Recall for ranking."""
import numpy as np

from matchzoo import engine


class Recall(engine.BaseMetric):
    """Precision metric."""

    ALIAS = 'recall'

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`RecallMetric` constructor.

        :param k: Number of results to consider.
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate recall@k.

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Precision @ k
        :raises: ValueError: len(r) must be >= k.
        """
        if self._k <= 0:
            raise ValueError('self._k must be larger than 0.')
        coupled_pair = engine.sort_and_couple(y_pred, y_true)
        recall = 0.0
        for idx, (score, label) in enumerate(coupled_pair):
            if idx >= self._k:
                break
            if score > self._threshold:
                recall += 1.
        return recall / self._k
