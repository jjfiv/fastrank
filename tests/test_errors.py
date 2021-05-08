from fastrank import clib
from fastrank.clib import CQRel
import pytest


def test_query_json_err():
    with pytest.raises(ValueError):
        message = clib.query_json("missing_query")
        print(message)


def test_not_qrel():
    with pytest.raises(ValueError):
        message = CQRel.load_file("../README.md")
        print(message)


def test_not_qrel_json():
    with pytest.raises(ValueError):
        message = CQRel.from_dict({"cat": "woof"})  # type:ignore
        print(message)


def test_qrel_missing_query():
    with pytest.raises(ValueError):
        qrel = CQRel.load_file("examples/newsir18-entity.qrel")
        print(qrel._query_json("MISSING_QUERY"))