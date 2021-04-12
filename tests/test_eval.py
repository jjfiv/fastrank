from fastrank.clib import evaluate_query
from sklearn.metrics import average_precision_score

scores = [0.9, 0.7, 0.5, 0.4]
truth = [0, 1, 0, 1]

rev_scores = list(reversed(scores))
rev_truth = list(reversed(truth))


def test_ap():
    sk_val = average_precision_score(y_score=scores, y_true=truth)
    fr_val = evaluate_query('ap', truth, scores)
    print(sk_val, fr_val)
    assert sk_val == fr_val

def test_rev_ap():
    sk_val = average_precision_score(y_score=rev_scores, y_true=rev_truth)
    fr_val = evaluate_query('ap', rev_truth, rev_scores)
    print(sk_val, fr_val)
    assert sk_val == fr_val

def test_mrr():
    assert 0.5 == evaluate_query("mrr", truth, scores)
    assert 1.0 == evaluate_query("mrr", [1,0,0,0], scores)
    assert 0.25 == evaluate_query("mrr", [1,0,0,0], rev_scores)