from dataclasses import dataclass

@dataclass
class Metrics:
    fps: float
    fns: float
    tps: float
    tns: float
    precision: float
    recall: float
    f_beta_score: float
    accuracy: float

def precision(tp: float, fp: float) -> float:
    """
    Precision = TP / (TP + FP)
    """
    denom = tp + fp

    if denom == 0:
        return 0.0

    return tp / denom


def recall(tp: float, fn: float) -> float:
    """
    Recall = TP / (TP + FN)
    """
    denom = tp + fn

    if denom == 0:
        return 0.0

    return tp / denom


def accuracy(tp: float, tn: float, fp: float, fn: float) -> float:
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    denom = tp + tn + fp + fn

    if denom == 0:
        return 0.0

    return (tp + tn) / denom


def f_beta_score(
    tp: float,
    fp: float,
    fn: float,
    beta: float = 1.0
) -> float:
    """
    General F-beta score.

    beta > 1:
        emphasizes recall

    beta < 1:
        emphasizes precision

    beta = 1:
        F1-score
    """

    p = precision(tp, fp)
    r = recall(tp, fn)

    if p == 0 and r == 0:
        return 0.0

    beta2 = beta ** 2

    return (
        (1 + beta2) * p * r
    ) / (
        beta2 * p + r
    )

