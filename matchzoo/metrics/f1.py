"""Recall for ranking."""
import numpy as np

from matchzoo import engine


class F1(engine.BaseMetric):
    """F1 metric."""

    ALIAS = 'f1'

    def __init__(self, k: int = 1, threshold: float = 0.5):
        """
        :class:`F1Metric` constructor.

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
        Calculate f1@k.

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.

        :return: f1 @ k
        :raises: ValueError: len(r) must be >= k.
        """
        fp = 0
        fn = 0
        tp = 0
        for label, score in zip(y_true, y_pred):
            if label <= 0:
                if score > self._threshold:
                    fp += 1
                else:
                    fn += 1
            elif score > self._threshold:
                tp += 1
        denominator = tp + 0.5 * (fp + fn)

        return tp / denominator if denominator > 0 else 0.0
