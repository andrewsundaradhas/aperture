"""Loop-closure before/after verification.

The wedge feature: after a retrain, re-measure the *same failure signature* and attribute the
task-success-rate delta to the original cluster. "Same failure signature" means the same task
context that used to fail — matched by the cluster's instruction set (falling back to the
dominant failure surface) — so a post-retrain batch of the same tasks is compared like-for-like.

  pre_success_rate  = success rate over the original cluster's episodes (the pre-retrain failures)
  post_success_rate = success rate over the post-retrain batch matching that signature
  delta             = post - pre   (positive == the fix worked)
"""

from __future__ import annotations

from dataclasses import dataclass

from aperture.core.models import Episode, FailureCluster


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _matches_signature(ep: Episode, instructions: set[str], dominant_surface: str | None) -> bool:
    if instructions and _norm(ep.instruction) in instructions:
        return True
    if not instructions and dominant_surface and ep.classification:
        return ep.classification.surface == dominant_surface
    return False


@dataclass
class VerificationResult:
    pre_success_rate: float
    post_success_rate: float
    pre_n: int
    post_n: int
    delta: float
    matched_episode_ids: list[str]


def compute_verification(
    cluster: FailureCluster,
    cluster_episodes: list[Episode],
    new_batch: list[Episode],
) -> VerificationResult:
    instructions = {_norm(e.instruction) for e in cluster_episodes if e.instruction}
    pre_n = len(cluster_episodes)
    pre_success = sum(1 for e in cluster_episodes if e.outcome == "success")
    pre_rate = pre_success / pre_n if pre_n else 0.0

    matched = [e for e in new_batch if _matches_signature(e, instructions, cluster.dominant_surface)]
    post_n = len(matched)
    post_success = sum(1 for e in matched if e.outcome == "success")
    post_rate = post_success / post_n if post_n else 0.0

    return VerificationResult(
        pre_success_rate=round(pre_rate, 4),
        post_success_rate=round(post_rate, 4),
        pre_n=pre_n,
        post_n=post_n,
        delta=round(post_rate - pre_rate, 4),
        matched_episode_ids=[e.id for e in matched],
    )
