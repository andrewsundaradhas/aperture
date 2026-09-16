"""The classifier benchmark has to stay runnable and honest.

Its value is that it can be re-run and disagreed with. These tests keep the generator from
quietly drifting into something that flatters the heuristics.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.run_eval import scores
from eval.synthetic import SURFACES, build_eval_set


@pytest.fixture(scope="module")
def eval_set():
    return build_eval_set(n_per_surface=60, seed=0)


def test_eval_set_is_balanced(eval_set):
    """Balanced support is what makes the majority-class baseline exactly 1/3."""
    counts = {s: sum(1 for e in eval_set if e.surface == s) for s in SURFACES}
    assert set(counts.values()) == {60}


def test_eval_set_is_deterministic():
    a = build_eval_set(n_per_surface=20, seed=7)
    b = build_eval_set(n_per_surface=20, seed=7)
    assert [e.surface for e in a] == [e.surface for e in b]
    assert [e.confidences for e in a] == [e.confidences for e in b]


def test_severity_is_swept_not_clustered(eval_set):
    """Every surface must span marginal to blatant, or per-severity accuracy means nothing."""
    for surface in SURFACES:
        sev = [e.severity for e in eval_set if e.surface == surface]
        assert min(sev) < 0.1 and max(sev) > 0.9


def test_some_episodes_drop_signals(eval_set):
    """Real logs lose sensors; an eval set where every channel is always present is too easy."""
    assert any(e.dropped for e in eval_set)


def test_generator_does_not_consult_the_heuristics():
    """The point of the set: it is built from a causal model, not from the trigger conditions.

    If `synthetic.py` ever imports the heuristics, the benchmark becomes circular.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "eval" / "synthetic.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)

    # Prose may discuss the heuristics; importing them is what would make it circular.
    offending = [name for name in imported if "heuristic" in name or "classifier" in name]
    assert not offending, f"the generator imports the code under test: {offending}"


def test_scores_match_a_hand_computed_case():
    y_true = ["perception", "perception", "grounding", "motor"]
    y_pred = ["perception", "motor", "grounding", "motor"]
    result = scores(y_true, y_pred)

    assert result["accuracy"] == 0.75
    # perception: tp=1 fp=0 fn=1 -> P 1.0, R 0.5
    assert result["per_class"]["perception"]["precision"] == 1.0
    assert result["per_class"]["perception"]["recall"] == 0.5
    # motor: tp=1 fp=1 fn=0 -> P 0.5, R 1.0
    assert result["per_class"]["motor"]["precision"] == 0.5
    assert result["per_class"]["motor"]["recall"] == 1.0


def _predict(episodes):
    from aperture.evaluation.classifier import classify_frames
    from aperture.ingestion.schemas import NormalizedFrame

    out = []
    for e in episodes:
        frames = [
            NormalizedFrame(t=t, action_confidence=c, contact_force=f, subgoal=s)
            for t, (c, f, s) in enumerate(zip(e.confidences, e.forces, e.subgoals))
        ]
        out.append(classify_frames(frames).surface)
    return out


def test_heuristics_beat_the_majority_baseline(eval_set):
    """The claim the report exists to support."""
    y_true = [e.surface for e in eval_set]
    accuracy = scores(y_true, _predict(eval_set))["accuracy"]
    assert accuracy > 1 / 3, f"heuristics ({accuracy:.3f}) do not beat majority-class guessing"


def test_held_out_accuracy_does_not_regress():
    """A floor on the *held-out* draw — the seed the thresholds were never tuned against.

    Guards the two fixes that took held-out macro F1 from 0.855 to 0.887: the peak-anchored
    confidence baseline, and the swept thresholds. Set below the measured value so ordinary
    noise does not trip it, but high enough that losing either fix would.
    """
    episodes = build_eval_set(n_per_surface=200, seed=99)
    result = scores([e.surface for e in episodes], _predict(episodes))
    assert result["macro_f1"] > 0.80, f"held-out macro F1 regressed to {result['macro_f1']:.3f}"


def test_no_class_is_left_behind():
    """An aggregate number can hide one broken class — perception sat at F1 0.579 while overall
    accuracy read 0.748. Every class has to carry its weight."""
    episodes = build_eval_set(n_per_surface=200, seed=99)
    result = scores([e.surface for e in episodes], _predict(episodes))
    weak = {s: v["f1"] for s, v in result["per_class"].items() if v["f1"] < 0.70}
    assert not weak, f"class(es) below F1 0.70: {weak}"
