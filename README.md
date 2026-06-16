# LLMs on ConTRoL

**Can Prompt Engineering Bridge the Long-Context NLI Reasoning Gap?**

Replicates and extends Liu et al. (2020) by evaluating modern open-source LLMs on the ConTRoL dataset using three prompting strategies — without any fine-tuning.

---

## Dataset

[ConTRoL](https://github.com/csitfun/ConTRoL-dataset) — 8,325 expert-designed premise-hypothesis pairs requiring complex contextual reasoning over long passages. Labels: Entailment, Contradiction, Neutral.

Place the dataset at `data/ConTRoL-dataset/` (already gitignored).

---

## Experiments

| Experiment | Description |
|---|---|
| `zero_shot` | Full passage + hypothesis → label |
| `few_shot_cot` | Three fixed examples (one per label) prepended to the zero-shot prompt |
| `hdqd_pipeline` | Two-stage: hypothesis → sub-questions → evidence-grounded verdict |

---

## Models

All models run via the lab API (Tailscale):

`llama3.1-8b`, `llama3.1-70b`, `qwen2.5-32b`, `qwen2.5-72b`, `qwen3.6-35b`, `qwen3.6-35b-ablit`, `gpt-oss-20b`, `gemma4-31b`, `nemotron-70b`

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env  # then fill in LAB_API_KEY
```

---

## Usage

```bash
# Run an experiment
python main.py --experiment zero_shot --model llama3.1-8b

# Test with a few samples first
python main.py --experiment zero_shot --model llama3.1-8b --limit 3
```

Results are saved to `results/{experiment}_{model}.json`.
Logs are saved to `logs/{experiment}_{model}_{timestamp}.log`.

---

## Project Structure

```
├── config/         # API settings, model list, logger, NLIOutput schema
├── data/           # Data loading (train + test splits)
├── prompts/        # Prompt templates (system + user)
├── runner/         # Experiment runners (one per experiment)
├── results/        # Output JSON files (gitignored)
├── logs/           # Run logs (gitignored)
├── open_source_llm.py  # LangChain wrapper for lab API
└── main.py         # Entry point
```

---

## References

- Liu et al. (2020). *Natural Language Inference in Context.* arXiv:2011.04864
- Wei et al. (2022). *Chain-of-Thought Prompting.* NeurIPS
- Kintsch (1988). *Construction-Integration Model.* Psychological Review
- Graesser et al. (1994). *Constructing Inferences.* Psychological Review
