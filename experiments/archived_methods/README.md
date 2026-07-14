# Archived methods

`hdqd_pipeline`, `q2_pipeline`, and `h_multihop` — retired from the active
method roster (`main.py`'s `RUNNERS`, `compare_question_methods.py`). The
active set is: `zero_shot`, `few_shot_cot`, `p_question`, `h_question`,
`bridge_question`, `retrieve_then_classify`.

Code is preserved here, not deleted, in case any of it is useful again later.
Each runner still imports its own prompts from
`experiments.archived_methods.prompts.*` (updated on the move) and still
imports shared components (`prompts.shared_classifier`,
`prompts.shared_answering`, `utils.locator_extractor`, etc.) from their
original locations, which are unchanged. Not wired into `main.py` or any
other active entry point — importing `experiments.archived_methods.runner.*`
directly still works, but nothing currently does.
