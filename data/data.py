import json
import os
from config.config import DATA_PATH

LABEL_MAP = {
    "e": "Entailment",
    "n": "Neutral",
    "c": "Contradiction",
}


def load_split(split: str) -> list[dict]:
    path = os.path.join(DATA_PATH, "data", f"{split}.jsonl")
    samples = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            samples.append({
                "id": raw["uid"],
                "premise": raw["premise"],
                "hypothesis": raw["hypothesis"],
                "label": LABEL_MAP[raw["label"]],
            })
    return samples


def load_data() -> list[dict]:
    return load_split("train") + load_split("test")
