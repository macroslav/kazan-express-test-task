from typing import Sequence


def hierarchical_f1_score(predicts: Sequence, true: Sequence) -> float:
    """ Count hierarchical f1 score

    Args:
        predicts: Sequence of predicted paths
        true: Sequence of true paths

    Returns:
        float: hierarchical f1 score
    """
    total_prec_denominator = 0
    total_rec_denominator = 0
    total_rec_numerator = 0
    total_prec_numerator = 0
    for predicted_path, true_path in zip(predicts, true):
        prec_numerator, prec_denominator = _count_hierarchical_precision(predicted_path, true_path)
        rec_numerator, rec_denominator = _count_hierarchical_recall(predicted_path, true_path)
        total_prec_denominator += prec_denominator
        total_rec_denominator += rec_denominator
        total_rec_numerator += rec_numerator
        total_prec_numerator += prec_numerator

    h_precision = total_prec_numerator / total_prec_denominator
    h_recall = total_rec_numerator / total_rec_denominator

    return 2 * h_precision * h_recall / (h_precision + h_recall)


def _count_hierarchical_precision(predicted_labels, true_labels) -> tuple[int, int]:
    """ Count hierarchical precision """
    return _count_intersection_size(predicted_labels, true_labels), len(predicted_labels)


def _count_hierarchical_recall(predicted_labels, true_labels) -> tuple[int, int]:
    """ Count hierarchical recall """
    return _count_intersection_size(predicted_labels, true_labels), len(true_labels)


def _count_intersection_size(first_sequence: Sequence, second_sequence: Sequence) -> int:
    """ Count intersection size of two sequences: |a ∩ b| """
    return len(list(set(first_sequence) & set(second_sequence)))
