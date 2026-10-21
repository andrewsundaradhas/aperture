"""Build the sample datasets offered on the dashboard's Upload page.

Both are real public data, unmodified apart from being cut down to a downloadable size:

* ``xarm-lift-40-episodes.zip`` — the first 40 episodes of ``lerobot/xarm_lift_medium``
  (Hugging Face, MIT), in its native LeRobot v3 layout (parquet + mp4). Policy rollouts of a
  "pick up the cube and lift it" task in simulation, a mix of successes and failures — the sample
  that exercises failure analysis end to end.
* ``nyu-rot-xarm-real-robot.tfrecord`` — the ``nyu_rot_dataset`` shard from Open X-Embodiment
  (Google Cloud Storage, CC BY 4.0): 14 demonstrations recorded on a real xArm, in RLDS. Human
  demonstrations, so every episode succeeds; it shows real-robot RLDS ingestion.
* ``signal-logged-demo.tfrecord`` — **synthetic**. 15 episodes in RLDS with the extra per-step
  channels a deployed policy can log (action confidence, contact force, sub-goal). Public
  datasets carry none of these, so this is the sample that shows every failure cause
  (perception, instruction grounding, motor) being identified.

Usage (needs the API's ``[datasets]`` extra for pyarrow + av)::

    apps/api/.venv/bin/python scripts/build_samples.py [--xarm-dir PATH] [--out DIR]

``--xarm-dir`` is a local copy of ``lerobot/xarm_lift_medium``; without it the script downloads
the three files it needs from Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

N_EPISODES = 40
HF = "https://huggingface.co/datasets/lerobot/xarm_lift_medium/resolve/main"
XARM_FILES = [
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
    "videos/observation.image/chunk-000/file-000.mp4",
]
NYU_SHARD = (
    "https://storage.googleapis.com/gresearch/robotics/nyu_rot_dataset_converted_externally_to_rlds/"
    "0.1.0/nyu_rot_dataset_converted_externally_to_rlds-train.tfrecord-00000-of-00001"
)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _cut_video(src: Path, dest: Path, end_seconds: float) -> None:
    """Copy the frames that start before `end_seconds`, without re-encoding.

    The cut starts at t=0 (a keyframe), and the boundary is exact: the reader requires one video
    frame per parquet row, so a frame more or less makes the sample unreadable.
    """
    import av

    dest.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as inp, av.open(str(dest), "w") as out:
        istream = inp.streams.video[0]
        ostream = out.add_stream_from_template(istream, opaque=True)  # copy, never decode
        for packet in inp.demux(istream):
            if packet.dts is None:
                continue
            if packet.pts is not None and float(packet.pts * istream.time_base) >= end_seconds:
                break
            packet.stream = ostream
            out.mux(packet)


# --- RLDS writer (TFRecord framing + tf.train.Example), enough for the signal demo -----------

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _floats(values) -> bytes:
    import struct

    return _ld(2, _ld(1, b"".join(struct.pack("<f", float(v)) for v in values)))


def _bytes(values) -> bytes:
    return _ld(1, b"".join(_ld(1, v) for v in values))


def _example(features: dict) -> bytes:
    entries = b"".join(_ld(1, _ld(1, k.encode()) + _ld(2, v)) for k, v in features.items())
    return _ld(1, entries)


def _tfrecord(payloads) -> bytes:
    import struct
    import zlib

    def masked(data: bytes) -> int:  # CRC32C is expected by TF; our reader does not verify it
        crc = zlib.crc32(data) & 0xFFFFFFFF
        return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF

    out = bytearray()
    for p in payloads:
        header = struct.pack("<Q", len(p))
        out += header + struct.pack("<I", masked(header)) + p + struct.pack("<I", masked(p))
    return bytes(out)


def build_signals_demo(out: Path) -> dict:
    """Synthetic RLDS episodes that carry policy confidence, contact force and sub-goals."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))
    from aperture import fixtures as fx
    from aperture.ingestion.normalize import normalize

    plan = [
        (fx.perception_failure_rlds, "pick up the red block"),
        (fx.perception_failure_rlds, "pick up the glass bottle under the window glare"),
        (fx.perception_failure_rlds, "grasp the transparent cup"),
        (fx.perception_failure_rlds, "place the bolt in the dark bin"),
        (fx.grounding_failure_rlds, "move object to the shelf"),
        (fx.grounding_failure_rlds, "hand me the other wrench"),
        (fx.grounding_failure_rlds, "put the left box on the pallet"),
        (fx.grounding_failure_rlds, "move the small part to the tray"),
        (fx.motor_failure_lerobot, "insert the peg"),
        (fx.motor_failure_lerobot, "tighten the connector"),
        (fx.motor_failure_lerobot, "press-fit the bearing"),
        (fx.motor_failure_lerobot, "seat the battery pack"),
        (fx.success_rlds, "pick up the red block"),
        (fx.success_rlds, "move object to the shelf"),
        (fx.success_rlds, "insert the peg"),
    ]
    payloads = []
    for make, task in plan:
        raw = make(task)
        name = "e.lerobot.json" if make is fx.motor_failure_lerobot else "e.rlds.json"
        ep = normalize(raw, name)
        n = len(ep.frames)
        ok = ep.outcome == "success"
        feats = {
            "steps/language_instruction": _bytes([task.encode()] * n),
            "steps/action": _floats([v for f in ep.frames for v in (f.action or [0.0] * 4)]),
            "steps/reward": _floats([1.0 if (ok and t == n - 1) else 0.0 for t in range(n)]),
            "steps/action_confidence": _floats([f.action_confidence or 0.0 for f in ep.frames]),
            "steps/contact_force": _floats([f.contact_force or 0.0 for f in ep.frames]),
        }
        if any(f.subgoal for f in ep.frames):
            feats["steps/subgoal"] = _bytes([(f.subgoal or "").encode() for f in ep.frames])
        payloads.append(_example(feats))
    out.write_bytes(_tfrecord(payloads))
    return {"episodes": len(payloads)}


def info_fps(src: Path) -> float:
    return float(json.loads((src / XARM_FILES[0]).read_text())["fps"])


def build_xarm(src: Path, out_zip: Path) -> dict:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "xarm-lift-40-episodes"
        episodes = pq.read_table(src / XARM_FILES[3])
        episodes = episodes.filter(pc.less(episodes["episode_index"], N_EPISODES))
        last_row = max(episodes["dataset_to_index"].to_pylist())
        video_end = max(episodes["videos/observation.image/to_timestamp"].to_pylist())

        data = pq.read_table(src / XARM_FILES[4])
        data = data.filter(pc.less(data["episode_index"], N_EPISODES))
        assert data.num_rows == last_row, (data.num_rows, last_row)

        for rel, table in ((XARM_FILES[3], episodes), (XARM_FILES[4], data)):
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dst / rel)
        for rel in (XARM_FILES[1], XARM_FILES[2]):
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / rel, dst / rel)

        info = json.loads((src / XARM_FILES[0]).read_text())
        info.update(
            robot_type="xarm",  # the dataset leaves it "unknown"; it is a UFactory xArm
            total_episodes=N_EPISODES,
            total_frames=data.num_rows,
            splits={"train": f"0:{N_EPISODES}"},
        )
        (dst / XARM_FILES[0]).write_text(json.dumps(info, indent=2))

        # `to_timestamp` of the last kept episode is the first frame of the next one; stop half a
        # frame before it so float rounding cannot let that frame in.
        _cut_video(src / XARM_FILES[5], dst / XARM_FILES[5], video_end - 0.5 / info_fps(src))

        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(dst.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(dst.parent))
    return {"episodes": N_EPISODES, "frames": data.num_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--xarm-dir", type=Path, help="local copy of lerobot/xarm_lift_medium")
    ap.add_argument("--out", type=Path, default=Path("apps/web/public/samples"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        src = args.xarm_dir
        if src is None:
            src = Path(tmp) / "xarm"
            for rel in XARM_FILES:
                print("downloading", rel)
                _fetch(f"{HF}/{rel}", src / rel)
        stats = build_xarm(src, args.out / "xarm-lift-40-episodes.zip")
        print("xarm sample:", stats)

    print("signal demo:", build_signals_demo(args.out / "signal-logged-demo.tfrecord"))

    nyu = args.out / "nyu-rot-xarm-real-robot.tfrecord"
    print("downloading nyu_rot shard")
    _fetch(NYU_SHARD, nyu)
    for f in sorted(args.out.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
