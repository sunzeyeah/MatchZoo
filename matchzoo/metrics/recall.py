"""Recall for ranking."""
import numpy as np

from matchzoo import engine


class Recall(engine.BaseMetric):
    """Recall metric."""

    ALIAS = 'recall'

    def __init__(self, k: int = 1, threshold: float = 0.5):
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

        :return: Recall @ k
        :raises: ValueError: len(r) must be >= k.
        """
        t = 0
        tp = 0

        for label, score in zip(y_true, y_pred):
            if label > 0:
                t += 1
                if score > self._threshold:
                    tp += 1

        return tp / t if t > 0 else 0.0
