"""Unit tests for utils/question_selectors.py's select_for_voting."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.question_selectors import select_for_voting


def _q(text, qtype):
    return {"q": text, "type": qtype}


def test_keeps_top_cap_by_relevance_regardless_of_type():
    questions = [
        _q("Did X surpass Y in skill?", "relation"),
        _q("What is the capital of France?", "fact"),
        _q("How tall is the tower?", "relation"),
    ]
    result = select_for_voting(questions, "X surpassed Y in skill.", cap=1)
    assert len(result) == 1
    assert result[0]["q"] == "Did X surpass Y in skill?"


def test_relevant_fact_beats_irrelevant_relation():
    # A fact-type question that closely matches the hypothesis should
    # outrank a relation-type question about something unrelated — type no
    # longer gives relations automatic priority.
    questions = [
        _q("What is the weather today?", "relation"),
        _q("Did X surpass Y in skill?", "fact"),
    ]
    result = select_for_voting(questions, "X surpassed Y in skill.", cap=1)
    assert len(result) == 1
    assert result[0]["q"] == "Did X surpass Y in skill?"


def test_returns_all_when_under_cap():
    questions = [
        _q("How does X compare to Y?", "relation"),
        _q("What year did X happen?", "fact"),
    ]
    result = select_for_voting(questions, "X surpassed Y.", cap=10)
    assert len(result) == 2


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
