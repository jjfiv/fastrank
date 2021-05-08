from .clib import evaluate_query


def average_precision_score(y_score, y_true, qids) -> float:
    return evaluate_query("ap", y_true, y_score)


def ndcg_score(y_score, y_true, qids) -> float:
    return evaluate_query("ndcg", y_true, y_score)


def reciprocal_rank(y_score, y_true) -> float:
    return evaluate_query("mrr", y_true, y_score)
