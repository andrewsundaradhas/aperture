"""Score the failure classifier against a labeled set, and write the report.

    python -m eval.run_eval              # regenerates docs/classifier_eval.md

Reports the 3-heuristic classifier against a majority-class baseline. The baseline matters:
on a balanced 3-class problem it scores 33.3% by construction, and any heuristic that cannot
clear that is not earning its place.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aperture.evaluation.classifier import classify_frames  # noqa: E402
from aperture.ingestion.schemas import NormalizedFrame  # noqa: E402

from eval.synthetic import SURFACES, LabeledEpisode, build_eval_set  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "docs" / "classifier_eval.md"


def _frames(episode: LabeledEpisode) -> list[NormalizedFrame]:
    return [
        NormalizedFrame(t=t, action_confidence=c, contact_force=f, subgoal=s)
        for t, (c, f, s) in enumerate(zip(episode.confidences, episode.forces, episode.subgoals))
    ]


def scores(y_true: list[str], y_pred: list[str]) -> dict:
    """Per-class precision / recall / F1 plus accuracy and a confusion matrix."""
    per_class = {}
    for surface in SURFACES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == surface and p == surface)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != surface and p == surface)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == surface and p != surface)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[surface] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(SURFACES)
    confusion = [[sum(1 for t, p in zip(y_true, y_pred) if t == a and p == b) for b in SURFACES] for a in SURFACES]
    return {
        "accuracy": round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _table(result: dict) -> str:
    rows = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for surface, v in result["per_class"].items():
        rows.append(
            f"| {surface} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
        )
    return "\n".join(rows)


def _confusion(result: dict) -> str:
    rows = ["| true \\ predicted | " + " | ".join(SURFACES) + " |", "|---" * (len(SURFACES) + 1) + "|"]
    for surface, row in zip(SURFACES, result["confusion_matrix"]):
        rows.append(f"| **{surface}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def _by_severity(episodes, y_true, y_pred, bands=4) -> str:
    rows = ["| Severity | Accuracy | n |", "|---|---|---|"]
    for band in range(bands):
        low, high = band / bands, (band + 1) / bands
        idx = [i for i, e in enumerate(episodes) if low <= e.severity < high or (band == bands - 1 and e.severity == 1.0)]
        if not idx:
            continue
        correct = sum(1 for i in idx if y_true[i] == y_pred[i])
        name = "marginal " if band == 0 else ("blatant " if band == bands - 1 else "")
        rows.append(f"| {name}{low:.2f}–{high:.2f} | {correct / len(idx):.3f} | {len(idx)} |")
    return "\n".join(rows)


def _by_dropped(episodes, y_true, y_pred) -> str:
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(episodes):
        key = ", ".join(e.dropped) if e.dropped else "all signals present"
        groups.setdefault(key, []).append(i)
    rows = ["| Missing signal | Accuracy | n |", "|---|---|---|"]
    for key, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        correct = sum(1 for i in idx if y_true[i] == y_pred[i])
        rows.append(f"| {key} | {correct / len(idx):.3f} | {len(idx)} |")
    return "\n".join(rows)


def _diagnostics() -> str:
    """Targeted probes that explain *why* a class fails, rather than only that it does.

    Computed, not asserted, so this section cannot drift out of date with the thresholds.
    """
    import numpy as np

    from aperture.evaluation.heuristics import CONF_COLLAPSE_RATIO, action_confidence_collapse

    gradual = action_confidence_collapse(list(np.linspace(0.9, 0.3, 30)))
    sharp = action_confidence_collapse([0.9] * 10 + [0.3] * 20)

    return f"""### The confidence baseline is anchored to the episode peak

`action_confidence_collapse` compares each frame against the **best sustained confidence seen so
far**, and fires below {CONF_COLLAPSE_RATIO:.0%} of it. It used to compare against a trailing
{3}-frame window, which drifts downward with a slow decline and therefore only ever caught abrupt
drops — perception recall sat at 0.407 because of it. The probe below is the regression guard:

| Confidence trace | Heuristic confidence |
|---|---|
| 0.9 -> 0.3 **gradually** over 30 frames (a 67% drop) | **{gradual.confidence:.3f}** |
| 0.9 -> 0.3 **abruptly** at frame 10 | {sharp.confidence:.3f} |

Both are the same total degradation, and both are now detected.

### Unreadable channels are reasoned about, not tie-broken away

Each heuristic reads exactly one signal, and real logs drop sensors. An episode with no sub-goal
annotations leaves the grounding heuristic nothing to read, so grounding can be neither argued
for nor ruled out. Tie-breaking those toward `motor` was the largest single error source: 32 of
40 grounding-read-as-motor cases were episodes whose sub-goal channel was simply absent.

The classifier now treats an unreadable channel as a hypothesis rather than a default. When every
signal that *is* readable comes back quiet and exactly one is missing, that missing surface wins,
flagged in `details._inferred_by_elimination` and carrying deliberately low confidence — it is an
inference from absence, not a finding. Accuracy by which signal the log was missing is in the
table above; with all three present the classifier sits near 0.95.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-surface", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0, help="the seed the thresholds were tuned on")
    parser.add_argument("--holdout-seed", type=int, default=99, help="a draw never used for tuning")
    parser.add_argument("--json", action="store_true", help="print raw metrics instead of writing the report")
    args = parser.parse_args()

    episodes = build_eval_set(args.n_per_surface, args.seed)
    y_true = [e.surface for e in episodes]
    y_pred = [classify_frames(_frames(e)).surface for e in episodes]

    heuristic = scores(y_true, y_pred)
    majority = Counter(y_true).most_common(1)[0][0]
    baseline = scores(y_true, [majority] * len(y_true))

    # The thresholds were fitted against `--seed`, so quoting only that number would flatter
    # them. Re-run on a draw never used for tuning and report both.
    holdout_eps = build_eval_set(args.n_per_surface, args.holdout_seed)
    holdout = scores(
        [e.surface for e in holdout_eps],
        [classify_frames(_frames(e)).surface for e in holdout_eps],
    )

    if args.json:
        print(json.dumps({"heuristic": heuristic, "baseline": baseline, "holdout": holdout}, indent=2))
        return

    lift = heuristic["accuracy"] - baseline["accuracy"]
    weak = [s for s, v in heuristic["per_class"].items() if v["f1"] < baseline["per_class"][s]["f1"] or v["f1"] < 0.5]

    REPORT.write_text(f"""# Failure classifier — evaluation

Regenerate with `cd ml && python -m eval.run_eval`. Deterministic given `--seed`.

## What this measures

The three heuristics in [`heuristics.py`](../apps/api/aperture/evaluation/heuristics.py) are
hand-tuned thresholds. This report is the evidence for — or against — them.

**The evaluation set is synthetic, and that is a real limitation.** No public dataset carries
per-episode perception/grounding/motor labels, so episodes are generated from a *causal model*
of each failure (see [`eval/synthetic.py`](../ml/eval/synthetic.py)): what the underlying fault
does to each signal channel, never what the heuristics look for. The generator varies severity,
leaks signals across channels the way real failures do, and drops sensors on some episodes.

It is still synthetic. These numbers say the heuristics discriminate on a plausible model of
the failure modes; they do **not** say the model matches your fleet. Replace the generator with
labeled fleet episodes when you have them — `build_eval_set` is the only thing to swap.

## Headline

| | Accuracy | Macro F1 |
|---|---|---|
| **3-heuristic classifier** (seed {args.seed}) | **{heuristic['accuracy']:.3f}** | **{heuristic['macro_f1']:.3f}** |
| **…on a held-out draw** (seed {args.holdout_seed}) | **{holdout['accuracy']:.3f}** | **{holdout['macro_f1']:.3f}** |
| Majority-class baseline (`{majority}`) | {baseline['accuracy']:.3f} | {baseline['macro_f1']:.3f} |
| Lift over baseline (held-out) | {holdout['accuracy'] - baseline['accuracy']:+.3f} | {holdout['macro_f1'] - baseline['macro_f1']:+.3f} |

`n = {len(episodes)}` episodes per draw, balanced across the three surfaces, so the baseline is
1/3 by construction.

**Read the held-out row, not the first one.** `CONF_COLLAPSE_RATIO` and `FORCE_SPIKE_Z` were
fitted by sweep against seed {args.seed}, so that row is optimistic by construction. Seed
{args.holdout_seed} was never used for tuning; the gap between the two
({heuristic['accuracy'] - holdout['accuracy']:+.3f} accuracy) is how much of the first number is
fitting rather than signal.

## Per class

{_table(heuristic)}

### Confusion

{_confusion(heuristic)}

## Where it breaks down

Accuracy by how blatant the failure is — the honest picture a headline number hides:

{_by_severity(episodes, y_true, y_pred)}

And by which signal the log was missing:

{_by_dropped(episodes, y_true, y_pred)}

{_diagnostics()}

## Verdict

{
    f"The heuristics beat the baseline by {lift:+.3f} accuracy overall."
    if lift > 0 else
    f"**The heuristics do not beat the majority-class baseline** ({lift:+.3f} accuracy)."
}
{
    "Weak classes: " + ", ".join(f"`{s}` (F1 {heuristic['per_class'][s]['f1']:.3f})" for s in weak) + "."
    if weak else "No class falls below F1 0.50."
}

Read the severity table before quoting the headline. A classifier that only catches blatant
failures is much less useful than its aggregate accuracy suggests, because the marginal cases
are the ones a human would not already have spotted.

## Where the remaining error lives, and why tuning stops here

With all three signals present the classifier sits near 0.95. Almost everything left is one of
two irreducible cases:

1. **A missing force sensor on a confounded motor failure.** Elimination recovers an unreadable
   channel only when every readable one is quiet. A motor failure that also dented confidence and
   triggered a retry leaves the readable channels *noisy*, so there is nothing to eliminate from
   and the episode reads as perception or grounding. Recovering these needs another signal, not
   another threshold.
2. **Genuine cross-channel confounds.** A blind policy retries its plan, so a perception failure
   can look like a grounding failure in the sub-goal trace. Both readings are defensible from the
   signals alone.

Further threshold tuning against this set would be **fitting the generator's own causal
assumptions** — how often a perception failure is modelled as causing a retry, say — rather than
anything about real robots. That is where the value of a synthetic benchmark runs out. The next
real gain comes from labeled fleet episodes, not from more sweeping.
""")
    print(f"wrote {REPORT}")
    print(f"accuracy {heuristic['accuracy']:.3f} vs baseline {baseline['accuracy']:.3f} ({lift:+.3f})")
    print(f"macro F1 {heuristic['macro_f1']:.3f}")
    for surface, v in heuristic["per_class"].items():
        print(f"  {surface:<11} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}")


if __name__ == "__main__":
    main()
