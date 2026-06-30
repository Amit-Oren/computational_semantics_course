"""Unit tests for utils/pos_keyphrase.py and utils/json_parse.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.json_parse import parse_json
from utils.pos_keyphrase import extract_keyphrases


# ── json_parse tests ──────────────────────────────────────────────────────────

def test_parse_json_clean():
    raw = '{"questions": ["What happened?", "Who did it?"]}'
    result = parse_json(raw, ["questions"])
    assert result is not None
    assert result["questions"] == ["What happened?", "Who did it?"]


def test_parse_json_with_markdown_fence():
    raw = '```json\n{"indices": [0, 3, 7]}\n```'
    result = parse_json(raw, ["indices"])
    assert result is not None
    assert result["indices"] == [0, 3, 7]


def test_parse_json_with_prose():
    raw = 'Here is the answer:\n{"label": "Entailment"}\nHope that helps!'
    result = parse_json(raw, ["label"])
    assert result is not None
    assert result["label"] == "Entailment"


def test_parse_json_missing_key():
    raw = '{"answer": "Some answer"}'
    result = parse_json(raw, ["label"])
    assert result is None


def test_parse_json_no_json():
    raw = "Sorry, I cannot answer that."
    result = parse_json(raw, ["label"])
    assert result is None


def test_parse_json_malformed():
    raw = '{"label": "Entailment"'  # unterminated
    result = parse_json(raw, ["label"])
    assert result is None


# ── pos_keyphrase tests ───────────────────────────────────────────────────────

def test_extract_keyphrases_returns_dict():
    result = extract_keyphrases("The unemployment rate increased by 5%.")
    assert "keyphrases" in result
    assert "claim_from_H" in result
    assert isinstance(result["keyphrases"], list)


def test_extract_keyphrases_claim_from_H():
    hyp = "Workers earned higher wages in the 20th century."
    result = extract_keyphrases(hyp)
    assert result["claim_from_H"] == hyp.strip()


def test_extract_keyphrases_finds_nouns():
    result = extract_keyphrases("The unemployment rate increased significantly.")
    kp_lower = [k.lower() for k in result["keyphrases"]]
    assert any("rate" in k or "unemployment" in k for k in kp_lower)


def test_extract_keyphrases_finds_verb():
    result = extract_keyphrases("Wages increased by 10 percent.")
    kp_lower = [k.lower() for k in result["keyphrases"]]
    assert any("increased" in k for k in kp_lower)


def test_extract_keyphrases_no_duplicates():
    result = extract_keyphrases("The rate of the rate was high.")
    kp_lower = [k.lower() for k in result["keyphrases"]]
    assert len(kp_lower) == len(set(kp_lower))


if __name__ == "__main__":
    # Run without pytest
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
