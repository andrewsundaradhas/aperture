"""Counterfactual probing.

For an episode classified as a language-grounding failure, we re-run the *same visual input*
through the policy with controlled rephrasings of the instruction (synonym swap, reordering,
removing a modifier) and record whether the predicted target changes. A target that flips
under trivial rephrasings is evidence of a true grounding failure rather than a one-off.

Production runs the rephrasings through the customer's own open policy checkpoint (OpenVLA,
etc.) entirely offline — no paid LLM calls, since we are probing *their* policy, not calling
Claude/GPT. This module provides:

  * `generate_rephrasings` — the controlled edits (deterministic, no external calls).
  * a `PolicyProbe` protocol — the interface a real checkpoint implements.
  * `MockPolicyProbe` — a deterministic stand-in used locally and in tests, whose target
    prediction is sensitive to instruction wording in proportion to how ambiguous the
    instruction is, so the end-to-end sensitivity signal is exercisable without a GPU.
  * `probe_episode` — orchestrates the probe and returns the stored result shape.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

_MODIFIERS = {"red", "blue", "green", "small", "large", "left", "right", "top", "the", "a"}
_SYNONYMS = {
    "pick": "grab",
    "grab": "pick",
    "up": "off",
    "place": "put",
    "put": "place",
    "block": "cube",
    "cube": "block",
    "move": "shift",
}


def generate_rephrasings(instruction: str, n: int = 4) -> list[str]:
    """Return up to `n` controlled rephrasings: synonym swap, reordering, modifier removal."""
    tokens = instruction.strip().split()
    variants: list[str] = []

    # 1. synonym swap
    swapped = [_SYNONYMS.get(t.lower(), t) for t in tokens]
    if swapped != tokens:
        variants.append(" ".join(swapped))

    # 2. reorder (swap first two content words)
    if len(tokens) >= 2:
        reordered = tokens[:]
        reordered[0], reordered[1] = reordered[1], reordered[0]
        variants.append(" ".join(reordered))

    # 3. remove a modifier
    stripped = [t for t in tokens if t.lower() not in _MODIFIERS]
    if stripped and stripped != tokens:
        variants.append(" ".join(stripped))

    # 4. drop the last token
    if len(tokens) >= 2:
        variants.append(" ".join(tokens[:-1]))

    # de-dup, drop no-ops, cap at n
    seen = {instruction}
    out: list[str] = []
    for v in variants:
        if v and v not in seen:
            out.append(v)
            seen.add(v)
        if len(out) >= n:
            break
    return out


class PolicyProbe(Protocol):
    def predicted_target(self, instruction: str, visual_signature: str) -> str: ...


class MockPolicyProbe:
    """Deterministic stand-in for an open VLA checkpoint.

    `ambiguity` in [0, 1] controls how easily the predicted target flips under rephrasing —
    it is fed from the episode's grounding evidence so a genuinely ambiguous instruction
    produces an unstable target and a well-grounded one stays put. No randomness: identical
    inputs always give identical targets (required for reproducible probing).
    """

    def __init__(self, ambiguity: float) -> None:
        self.ambiguity = max(0.0, min(1.0, ambiguity))

    def predicted_target(self, instruction: str, visual_signature: str) -> str:
        h = hashlib.sha256(f"{visual_signature}".encode()).hexdigest()
        base_target = f"obj_{int(h[:4], 16) % 8}"
        if self.ambiguity < 0.34:
            # Well grounded: target is anchored to the scene, wording-invariant.
            return base_target
        # Ambiguous: wording perturbs the target, more so as ambiguity rises.
        wh = int(hashlib.sha256(instruction.encode()).hexdigest()[:8], 16)
        span = 1 + int(self.ambiguity * 7)
        return f"obj_{(int(h[:4], 16) + wh) % span}"


def probe_episode(
    instruction: str | None,
    visual_signature: str,
    probe: PolicyProbe,
    n: int = 4,
) -> dict:
    """Run the probe and summarize instruction sensitivity."""
    if not instruction:
        return {"applicable": False, "reason": "no language instruction on episode"}

    base = probe.predicted_target(instruction, visual_signature)
    rephrasings = generate_rephrasings(instruction, n=n)
    trials = []
    changed = 0
    for r in rephrasings:
        t = probe.predicted_target(r, visual_signature)
        flipped = t != base
        changed += int(flipped)
        trials.append({"instruction": r, "target": t, "changed": flipped})

    total = len(trials)
    sensitivity = changed / total if total else 0.0
    return {
        "applicable": True,
        "base_instruction": instruction,
        "base_target": base,
        "trials": trials,
        "sensitivity": round(sensitivity, 3),
        "verdict": "grounding-sensitive" if sensitivity >= 0.5 else "stable",
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
