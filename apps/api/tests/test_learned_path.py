"""End-to-end exercise of the learned path against real checkpoints.

Skipped unless `ml/models/{policy.pt,failure_head.pt}` exist and the `[ml]` extra is installed
— which is the state in CI and in a base install. Train them with
`python -m training.train` from `ml/` to make this suite run.

What it is here to catch: the heuristic path and the learned path are wired through the same
routers, so a shape/metadata regression in the learned branch is invisible to every other test.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from sqlalchemy import select

from aperture.core.models import Episode, Organization

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "ml" / "models"

pytestmark = pytest.mark.skipif(
    not ((MODEL_DIR / "policy.pt").exists() and (MODEL_DIR / "failure_head.pt").exists()),
    reason="no trained checkpoints in ml/models — run ml/training/train.py first",
)


@pytest.fixture(scope="module")
def learned_runtime():
    """Force the learned path on for this module, then restore the cached settings."""
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("sentence_transformers")

    from aperture.core.config import get_settings
    from aperture.ml import runtime

    get_settings.cache_clear()
    settings = get_settings()
    original_flag, original_dir = settings.use_learned_models, settings.local_model_dir
    settings.use_learned_models = True
    settings.local_model_dir = str(MODEL_DIR)

    if not runtime.is_available():
        pytest.skip("checkpoints present but not loadable in this environment")
    yield runtime

    settings.use_learned_models, settings.local_model_dir = original_flag, original_dir
    get_settings.cache_clear()


def _real_frame_png() -> bytes:
    """A real 84x84 training frame, encoded as PNG — not noise, which only proves shapes."""
    import numpy as np
    from PIL import Image

    data_root = REPO_ROOT / "ml" / "data" / "xarm_lift_medium"
    cache = data_root / ".cache" / "frames.npy"
    if cache.exists():
        frame = np.asarray(np.load(cache, mmap_mode="r")[0], dtype=np.uint8)
    else:  # dataset not downloaded — a synthetic scene still exercises every code path
        frame = np.zeros((84, 84, 3), dtype=np.uint8)
        frame[4:-4, 4:-4] = 150
        frame[40:50, 30:40] = (40, 120, 110)
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()


def _episode_with_image(png: bytes) -> bytes:
    img_b64 = base64.b64encode(png).decode()
    return json.dumps(
        {
            "embodiment_type": "xarm",
            "policy_name": "aperture-reference",
            "outcome": "fail",
            "steps": [
                {
                    "observation": {"action_confidence": 0.2, "contact_force": 1.0, "image": img_b64},
                    "language_instruction": "Pick up the cube and lift it.",
                    "subgoal": "reach",
                }
                for _ in range(3)
            ],
        }
    ).encode()


@pytest.fixture
def learned_episode(db_session, learned_runtime):
    from aperture.ingestion.normalize import normalize
    from aperture.ingestion.service import persist_episode

    org = db_session.execute(
        select(Organization).where(Organization.slug == "acme-robotics")
    ).scalar_one()
    ep = persist_episode(db_session, org, normalize(_episode_with_image(_real_frame_png()), "e.rlds.json"))
    db_session.commit()
    return db_session.get(Episode, ep.id)


def test_gateway_reports_learned_enabled(learned_runtime):
    from aperture.ml import gateway

    assert gateway.learned_enabled() is True


def test_classification_uses_the_learned_head(client, auth, learned_episode):
    body = client.post(f"/v1/episodes/{learned_episode.id}/classify", headers=auth).json()

    assert body["method"] == "learned", "an episode with an image must take the learned branch"
    assert body["surface"] in ["perception", "grounding", "motor"]
    probs = body["details"]["probs"]
    assert set(probs) == {"perception", "grounding", "motor"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
    # The reported confidence is the winning class's probability, not something else.
    assert body["confidence"] == pytest.approx(probs[body["surface"]], abs=1e-3)


def test_attention_heatmap_is_real_and_well_formed(client, auth, learned_episode):
    from aperture.core.storage import read_uri
    from aperture.ml.runtime import GRID

    body = client.post(f"/v1/episodes/{learned_episode.id}/attribution", headers=auth).json()
    assert body["attention_map_uri"]

    doc = json.loads(read_uri(body["attention_map_uri"]))
    assert doc["simulated"] is False, "a real heatmap must not be flagged simulated"
    assert doc["method"] == "cross_attention"
    assert doc["grid_size"] == GRID
    assert doc["frames"], "at least one frame image should have produced a grid"

    for frame in doc["frames"]:
        assert len(frame["grid"]) == GRID and all(len(row) == GRID for row in frame["grid"])
        total = sum(v for row in frame["grid"] for v in row)
        assert total == pytest.approx(1.0, abs=1e-2), "each grid is row-normalized to sum to 1"
        fx, fy = frame["focus"]
        assert 0 <= fx < GRID and 0 <= fy < GRID


def test_policy_predicts_a_finite_action(learned_runtime):
    import math

    from aperture.ml.model import ACTION_DIM

    mean, logvar = learned_runtime.predict_action(_real_frame_png(), "Pick up the cube and lift it.")
    assert len(mean) == ACTION_DIM and len(logvar) == ACTION_DIM
    assert all(math.isfinite(v) for v in mean + logvar), "NaN/inf means a broken checkpoint"


def test_visual_embedding_has_the_clustering_dimension(learned_runtime):
    import numpy as np

    emb = learned_runtime.embed_image(_real_frame_png())
    assert emb.shape == (384,), "clustering expects the 384-dim pooled ViT-Small embedding"
    assert np.isfinite(emb).all()


def test_two_different_scenes_embed_differently(learned_runtime):
    """A constant embedding would silently collapse every cluster into one."""
    import io as _io

    import numpy as np
    from PIL import Image

    base = np.zeros((84, 84, 3), dtype=np.uint8)
    base[4:-4, 4:-4] = 150

    def encode(img):
        buf = _io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    left, right = base.copy(), base.copy()
    left[40:50, 10:20] = (40, 120, 110)
    right[20:30, 60:70] = (40, 120, 110)

    a = learned_runtime.embed_image(encode(left))
    b = learned_runtime.embed_image(encode(right))
    assert not np.allclose(a, b, atol=1e-4)
