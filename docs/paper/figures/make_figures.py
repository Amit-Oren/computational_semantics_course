"""
Generates the two main-results bar charts for the paper from
scratchpad/master_results.json (produced by the accuracy-computation pass
over results/production/). Re-run any time the underlying result files change.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

with open("/private/tmp/claude-501/-Users-yuvalzohar-Documents-GitHub-computational-semantics-course/d889ed38-a059-4ae9-a45b-95a44aef6043/scratchpad/master_results.json") as f:
    master = json.load(f)

MODELS = ["qwen2.5-32b", "gpt-oss-20b", "gpt-4o-mini", "deepseek-v4-flash", "google-gemma-3-27b-it"]
MODEL_LABELS = {
    "qwen2.5-32b": "Qwen2.5-32B",
    "gpt-oss-20b": "gpt-oss-20B",
    "gpt-4o-mini": "GPT-4o-mini",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "google-gemma-3-27b-it": "Gemma-3-27B",
}
# fixed categorical order, palette.md slots 1-5 (validated adjacent-pair CVD-safe order)
COLORS = {
    "qwen2.5-32b": "#2a78d6",           # blue
    "gpt-oss-20b": "#eb6834",           # orange
    "gpt-4o-mini": "#1baf7a",           # aqua
    "deepseek-v4-flash": "#eda100",     # yellow
    "google-gemma-3-27b-it": "#e87ba4", # magenta
}

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": MUTED,
    "text.color": INK,
    "axes.labelcolor": SECONDARY_INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
})


def grouped_bar(ax, methods, method_labels, title, ymax=85):
    n_models = len(MODELS)
    bar_w = 0.8 / n_models
    x = range(len(methods))

    for i, model in enumerate(MODELS):
        vals = [master[m].get(model) for m in methods]
        offset = (i - (n_models - 1) / 2) * bar_w
        xs = [xi + offset for xi in x]
        ys = [v * 100 if v is not None else 0 for v in vals]
        # only draw bars where data actually exists
        xs_present = [xx for xx, v in zip(xs, vals) if v is not None]
        ys_present = [y for y, v in zip(ys, vals) if v is not None]
        ax.bar(xs_present, ys_present, width=bar_w * 0.88, color=COLORS[model],
               label=MODEL_LABELS[model], zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels([method_labels[m] for m in methods], fontsize=9)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="both", length=0)
    # chance-rate reference line (3-way classification)
    ax.axhline(33.3, color=MUTED, linewidth=1, linestyle=(0, (1, 2)), zorder=2)
    ax.text(len(methods) - 0.5, 33.3 + 1.2, "chance (33%)", fontsize=7.5, color=MUTED, ha="right")


# ─── Figure 1: six main methods, N=1500 ──────────────────────────────────────
main_methods = ["zero_shot", "few_shot_cot", "retrieve_then_classify",
                "h_question_pos", "h_question_srl", "bridge_question"]
main_labels = {
    "zero_shot": "zero-\nshot",
    "few_shot_cot": "few-\nshot",
    "retrieve_then_classify": "retrieve-\nthen-\nclassify",
    "h_question_pos": "h-question\n(pos)",
    "h_question_srl": "h-question\n(srl)",
    "bridge_question": "bridge-\nquestion",
}

fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
grouped_bar(ax, main_methods, main_labels,
            "Accuracy across the six N=1,500 methods", ymax=85)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("docs/paper/figures/main_results.pdf", bbox_inches="tight")
plt.close(fig)

# ─── Figure 2: three p_question variants, N=150 ──────────────────────────────
p_methods = ["p_question_decomposition", "p_question_seeded_pos", "p_question_seeded_srl"]
p_labels = {
    "p_question_decomposition": "p-question\n(decomposition)",
    "p_question_seeded_pos": "p-question\n(seeded, pos)",
    "p_question_seeded_srl": "p-question\n(seeded, srl)",
}

fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
grouped_bar(ax, p_methods, p_labels,
            "Accuracy on the three N=150 premise-question methods", ymax=85)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("docs/paper/figures/p_question_results.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved docs/paper/figures/main_results.pdf and p_question_results.pdf")
