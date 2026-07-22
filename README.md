# LLMs on ConTRoL

**Can Prompt Engineering Bridge the Long-Context NLI Reasoning Gap?**

Replicates and extends Liu et al. (2020) by evaluating modern open-source LLMs on the ConTRoL dataset using several question-based prompting pipelines — without any fine-tuning.

---

## Dataset

[ConTRoL](https://github.com/csitfun/ConTRoL-dataset) — 8,325 expert-designed premise-hypothesis pairs requiring complex contextual reasoning over long passages (train: 6,719, dev: 799, test: 805). Labels: Entailment, Contradiction, Neutral.

This is a separate repo, not tracked here — clone it into `data/ConTRoL-dataset/` (already gitignored):
```bash
git clone https://github.com/csitfun/ConTRoL-dataset.git data/ConTRoL-dataset
```

---

## Methods

| Method | Description |
|---|---|
| `zero_shot` | Full passage + hypothesis → label, single call, no decomposition |
| `few_shot_cot` | Same, with 3 fixed demonstrations (one per label) prepended |
| `retrieve_then_classify` | Locator runs directly on the hypothesis, no question generation — control baseline isolating the Locator's contribution |
| `h_question` | Probe questions generated from the hypothesis, answered from the premise |
| `p_question` | Questions generated from the premise (decomposed into facts/relations, or seeded), answered from the premise |
| `bridge_question` | Questions generated from both texts together — the only method whose Stage 1 sees premise and hypothesis at once |

See `docs/methods_overview.md` for full architecture details per method (stages, LLM call counts, aggregation/few-shot axes), and `summery.md` for ablation results.

---

## Models

Open-weight models run via a self-hosted lab server (Ollama-backed, reached over Tailscale):

`llama3.1-8b`, `llama3.1-70b`, `qwen2.5-32b`, `qwen2.5-72b`, `qwen3.6-35b`, `qwen3.6-35b-ablit`, `gpt-oss-20b`, `gemma4-31b`, `nemotron-70b`

Also configured: Hugging Face Inference API, Groq, and local Ollama — see `config/config.py`'s `MODELS` dict for the full routing table.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # then fill in LAB_API_URL and LAB_LLM_TOKEN
```

---

## Usage

```bash
# Run an experiment
python main.py --experiment zero_shot --model qwen2.5-32b

# Test with a few samples first
python main.py --experiment zero_shot --model qwen2.5-32b --limit 3

# p_question and h_question take extra flags — see main.py --help
python main.py --experiment p_question --model qwen2.5-32b --generation decomposition --limit 50
```

Results are saved to `results/{experiment}_{model}_{timestamp}.json`.
Logs are saved to `logs/{experiment}_{model}_{timestamp}.log`.

For ablation sweeps across multiple methods/seeders/aggregation modes, see `experiments/ablation_full.py`. For a full production sweep across all 9 method configurations on a large fixed sample, see `run_production_train.py`.

---

## Project Structure

```
├── config/           # API settings, model routing, structured-output wrapper, logger
├── data/             # Data loading (train/dev/test splits)
├── prompts/          # Prompt templates (system + user), per method
├── runner/           # Method orchestration — one module per method
├── utils/            # Shared helpers: Locator/Extractor, aggregation modes, retry,
│                     #   question selectors, sentence indexing, seeding
├── experiments/       # Ablation sweeps, method comparisons, one-off analysis scripts
│   └── archived/     # Superseded/unused code kept for reference
├── calibration/       # Timing/token-usage harness for estimating full-run costs
├── tests/             # Unit tests for deterministic, non-LLM utility code
├── notebooks/         # Colab-based alternate workflow
├── docs/              # Method architecture reference + design specs/plans
├── results/           # Output JSON files (gitignored)
│   └── production/    # Large-scale production-sweep results
├── logs/              # Run logs (gitignored)
├── run_production_train.py   # Full 9-method sweep on a fixed train-split sample
└── main.py            # Single-experiment entry point
```

---

## References

- Liu et al. (2020). *Natural Language Inference in Context.* arXiv:2011.04864
- Wei et al. (2022). *Chain-of-Thought Prompting.* NeurIPS
- Kintsch (1988). *Construction-Integration Model.* Psychological Review
- Graesser et al. (1994). *Constructing Inferences.* Psychological Review
