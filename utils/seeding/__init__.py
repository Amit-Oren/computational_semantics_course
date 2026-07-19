"""
Seeding module — interchangeable input-preprocessing strategies.

Each seeder takes a text string and returns a list of semantic units
(keyphrases, triples, or role structures) that seed the downstream
question-generation LLM call.

Unified interface:
    seeder = get_seeder("pos" | "svo" | "srl")
    seeds: list[str] = seeder.seed(text)
"""

from __future__ import annotations

from typing import Optional

from .pos import POSSeeder
from .svo import SVOSeeder
from .srl import SRLSeeder

SEEDERS: dict[str, type] = {
    "pos": POSSeeder,
    "svo": SVOSeeder,
    "srl": SRLSeeder,
}


def get_seeder(name: str, model: Optional[str] = None, params: Optional[dict] = None):
    """Instantiate a seeder by name.

    model/params are forwarded to seeders that make LLM calls (srl).
    Raises ValueError for unknown names.
    """
    if name not in SEEDERS:
        raise ValueError(f"Unknown seeder '{name}'; choose from {sorted(SEEDERS)}")
    return SEEDERS[name](model=model, params=params)
