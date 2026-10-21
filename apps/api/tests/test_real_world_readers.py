"""Reader behaviour on the shapes real robot datasets actually take.

Found by running real public datasets through ingestion: teleoperated LeRobot datasets with no
reward column could not be read at all, multi-camera recordings broke frame alignment, an explicit
`next.success` flag was ignored, RLDS 0/1 success rewards one step before the end were missed,
multi-part actions (Open X-Embodiment `action/world_vector`, ...) came through empty, and cameras
not called `image` were dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from tests._tfrecord_writer import bytes_list, example, float_list, tfrecord  # noqa: E402


def _video(path: Path, n: int, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = av.open(str(path), mode="w")
    s = c.add_stream("libx264", rate=15)
    s.width = s.height = size
    s.pix_fmt = "yuv420p"
    for i in range(n):
        c.mux(s.encode(av.VideoFrame.from_ndarray(np.full((size, size, 3), i % 256, np.uint8), format="rgb24")))
    c.mux(s.encode())
    c.close()


def _lerobot(root: Path, *, cams=("observation.image",), reward=None, success=None, eps=2, per=5) -> Path:
    n = eps * per
    feats = {c: {"dtype": "video"} for c in cams}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 15, "robot_type": "so101", "features": feats}))
    cols = {
        "episode_index": [e for e in range(eps) for _ in range(per)],
        "frame_index": [t for _ in range(eps) for t in range(per)],
        "action": [[0.1 * t, 0.2] for _ in range(eps) for t in range(per)],
        "observation.state": [[0.0, 1.0] for _ in range(n)],
    }
    if reward is not None:
        cols["next.reward"] = reward
    if success is not None:
        cols["next.success"] = success
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(cols), root / "data" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": list(range(eps)), "tasks": [["pick and place"]] * eps}),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    for c in cams:
        _video(root / "videos" / c / "chunk-000" / "file-000.mp4", n)
    return root


def test_teleop_dataset_without_reward_is_readable_and_marked_as_demonstrations(tmp_path):
    from aperture.ingestion.lerobot_v3 import to_normalized_episodes

    eps = to_normalized_episodes(_lerobot(tmp_path / "d", cams=("observation.images.up", "observation.images.side")))
    assert len(eps) == 2
    assert all(e.outcome == "success" and e.outcome_source == "none" for e in eps)
    assert eps[0].embodiment_type == "so101"
    assert all(f.image_b64 for f in eps[0].frames), "one camera's frames must align with the rows"


def test_explicit_success_flag_beats_reward(tmp_path):
    from aperture.ingestion.lerobot_v3 import to_normalized_episodes

    reward = [0.0] * 10
    success = [False] * 4 + [True] + [False] * 5  # episode 0 reaches success, episode 1 never does
    eps = to_normalized_episodes(_lerobot(tmp_path / "d", reward=reward, success=success))
    assert [e.outcome for e in eps] == ["success", "fail"]
    assert {e.outcome_source for e in eps} == {"flag"}


def test_images_are_capped_per_episode_but_every_frame_keeps_actions(tmp_path):
    from aperture.ingestion.lerobot_v3 import to_normalized_episodes

    eps = to_normalized_episodes(_lerobot(tmp_path / "d", eps=1, per=20), images_per_episode=4)
    frames = eps[0].frames
    assert len(frames) == 20 and all(f.action for f in frames)
    with_image = [f.t for f in frames if f.image_b64]
    assert len(with_image) == 4 and with_image[0] == 0 and with_image[-1] == 19


def _rlds(**steps) -> bytes:
    return tfrecord([example(steps)])


def test_rlds_success_reward_before_the_last_step_counts():
    from aperture.ingestion.normalize import normalize_many

    raw = _rlds(**{"steps/reward": float_list([0, 0, 1, 0]), "steps/action": float_list([0.0] * 8)})
    (ep,) = normalize_many(raw, "x.tfrecord")
    assert ep.outcome == "success" and ep.outcome_source == "reward"


def test_rlds_multipart_actions_and_unusual_camera_names():
    from aperture.ingestion.normalize import normalize_many

    raw = _rlds(**{
        "steps/action/world_vector": float_list([1, 2, 3, 4, 5, 6]),
        "steps/action/gripper_closedness_action": float_list([0.5, 0.7]),
        "steps/observation/hand_image": bytes_list([b"\x89PNG-a", b"\x89PNG-b"]),
        "steps/reward": float_list([0, 1]),
    })
    (ep,) = normalize_many(raw, "jaco_play-train.tfrecord-00102-of-00128")
    assert [f.action for f in ep.frames] == [pytest.approx([0.5, 1, 2, 3]), pytest.approx([0.7, 4, 5, 6])]
    assert all(f.image_b64 for f in ep.frames)
    assert ep.embodiment_type == "jaco_play"


def test_rlds_reads_logged_subgoals():
    from aperture.ingestion.normalize import normalize_many

    raw = _rlds(**{"steps/subgoal": bytes_list([b"reach", b"grasp", b"reach"]), "steps/reward": float_list([0, 0, 0])})
    (ep,) = normalize_many(raw, "x.tfrecord")
    assert [f.subgoal for f in ep.frames] == ["reach", "grasp", "reach"]


def test_outcome_override_labels_every_episode():
    from aperture.ingestion.normalize import IngestionError, normalize_many

    raw = _rlds(**{"steps/action": float_list([0.0, 1.0])})  # no reward: no outcome signal
    (auto,) = normalize_many(raw, "x.tfrecord")
    assert auto.outcome_source == "none"
    (forced,) = normalize_many(raw, "x.tfrecord", outcome="fail")
    assert forced.outcome == "fail" and forced.outcome_source == "override"
    with pytest.raises(IngestionError):
        normalize_many(raw, "x.tfrecord", outcome="maybe")


def test_upload_outcome_field_and_verify_refuses_unlabelled_runs(client, auth):
    from aperture import fixtures as fx
    from aperture.jobs.worker import drain

    raw = _rlds(**{"steps/action": float_list([0.0, 1.0, 2.0, 3.0])})
    r = client.post("/v1/episodes/upload", headers=auth, data={"outcome": "fail"},
                    files=[("files", ("runs.tfrecord", raw))])
    assert r.status_code == 201
    ep = client.get(f"/v1/episodes/{r.json()['episode_ids'][0]}", headers=auth).json()
    assert ep["outcome"] == "fail"

    for i in range(4):
        client.post("/v1/episodes/upload", headers=auth, files=[("files", (f"g{i}.rlds.json", fx.grounding_failure_rlds("stack cups")))])
    drain()
    client.post("/v1/clusters/recompute?wait=true", headers=auth)
    clusters = client.get("/v1/clusters", headers=auth).json()
    assert clusters
    v = client.post(f"/v1/clusters/{clusters[0]['id']}/verify", headers=auth, files=[("files", ("post.tfrecord", raw))])
    assert v.status_code == 422 and "don't record whether each run succeeded" in v.text
