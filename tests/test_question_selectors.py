"""Unit tests for utils/question_selectors.py's select_for_voting."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.question_selectors import select_for_voting


def _q(text, qtype):
    return {"q": text, "type": qtype}


def test_all_relations_kept_when_under_cap():
    questions = [
        _q("How does X compare to Y?", "relation"),
        _q("What year did X happen?", "fact"),
        _q("What year did Y happen?", "fact"),
    ]
    result = select_for_voting(questions, "X surpassed Y.", cap=10)
    types = [d["type"] for d in result]
    assert types.count("relation") == 1
    assert len(result) == 3  # nothing to trim, all fit under cap


def test_facts_fill_remaining_slots_by_relevance():
    questions = [
        _q("How does X compare to Y?", "relation"),
        _q("What is the capital of France?", "fact"),
        _q("Did X surpass Y in skill?", "fact"),
    ]
    result = select_for_voting(questions, "X surpassed Y in skill.", cap=2)
    assert len(result) == 2
    qs = [d["q"] for d in result]
    assert "How does X compare to Y?" in qs
    assert "Did X surpass Y in skill?" in qs  # more relevant fact kept over the unrelated one


def test_relations_trimmed_by_relevance_when_over_cap():
    questions = [
        _q("Did X surpass Y in skill?", "relation"),
        _q("What is the weather today?", "relation"),
        _q("How tall is the tower?", "relation"),
    ]
    result = select_for_voting(questions, "X surpassed Y in skill.", cap=1)
    assert len(result) == 1
    assert result[0]["q"] == "Did X surpass Y in skill?"


def test_empty_input_returns_empty_list():
    assert select_for_voting([], "Any hypothesis.", cap=10) == []


def test_cap_zero_returns_empty_list():
    questions = [_q("Some question?", "fact")]
    assert select_for_voting(questions, "Any hypothesis.", cap=0) == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
