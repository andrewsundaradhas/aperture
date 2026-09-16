# Failure classifier — evaluation

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
| **3-heuristic classifier** (seed 0) | **0.926** | **0.926** |
| **…on a held-out draw** (seed 99) | **0.920** | **0.920** |
| Majority-class baseline (`motor`) | 0.333 | 0.167 |
| Lift over baseline (held-out) | +0.587 | +0.753 |

`n = 1200` episodes per draw, balanced across the three surfaces, so the baseline is
1/3 by construction.

**Read the held-out row, not the first one.** `CONF_COLLAPSE_RATIO` and `FORCE_SPIKE_Z` were
fitted by sweep against seed 0, so that row is optimistic by construction. Seed
99 was never used for tuning; the gap between the two
(+0.006 accuracy) is how much of the first number is
fitting rather than signal.

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| perception | 0.936 | 0.910 | 0.923 | 400 |
| grounding | 0.885 | 0.940 | 0.911 | 400 |
| motor | 0.961 | 0.927 | 0.944 | 400 |

### Confusion

| true \ predicted | perception | grounding | motor |
|---|---|---|---|
| **perception** | 364 | 32 | 4 |
| **grounding** | 13 | 376 | 11 |
| **motor** | 12 | 17 | 371 |

## Where it breaks down

Accuracy by how blatant the failure is — the honest picture a headline number hides:

| Severity | Accuracy | n |
|---|---|---|
| marginal 0.00–0.25 | 0.953 | 300 |
| 0.25–0.50 | 0.903 | 300 |
| 0.50–0.75 | 0.933 | 300 |
| blatant 0.75–1.00 | 0.913 | 300 |

And by which signal the log was missing:

| Missing signal | Accuracy | n |
|---|---|---|
| all signals present | 0.956 | 932 |
| subgoal | 0.885 | 104 |
| contact_force | 0.837 | 98 |
| action_confidence | 0.902 | 41 |
| contact_force, subgoal | 0.357 | 14 |
| contact_force, action_confidence | 0.500 | 6 |
| subgoal, action_confidence | 0.200 | 5 |

### The confidence baseline is anchored to the episode peak

`action_confidence_collapse` compares each frame against the **best sustained confidence seen so
far**, and fires below 75% of it. It used to compare against a trailing
3-frame window, which drifts downward with a slow decline and therefore only ever caught abrupt
drops — perception recall sat at 0.407 because of it. The probe below is the regression guard:

| Confidence trace | Heuristic confidence |
|---|---|
| 0.9 -> 0.3 **gradually** over 30 frames (a 67% drop) | **0.639** |
| 0.9 -> 0.3 **abruptly** at frame 10 | 0.778 |

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


## Verdict

The heuristics beat the baseline by +0.593 accuracy overall.
No class falls below F1 0.50.

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
