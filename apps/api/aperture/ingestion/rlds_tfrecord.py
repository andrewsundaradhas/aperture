"""RLDS / Open X-Embodiment TFRecord reader — no TensorFlow required.

Why hand-rolled: pulling in `tensorflow` (or `tensorflow-datasets`) to read a container format
and one small protobuf would add roughly a gigabyte of dependency, pin a CUDA matrix, and make
the API impossible to install on a free dyno. Both formats involved are small and frozen:

* **TFRecord framing** — per record: `uint64` length, `uint32` masked CRC32C of the length
  bytes, the payload, `uint32` masked CRC32C of the payload. Little-endian throughout.
* **`tf.train.Example`** — `Example{ Features features = 1 }`,
  `Features{ map<string, Feature> feature = 1 }`, and a `Feature` is exactly one of
  `bytes_list = 1`, `float_list = 2`, `int64_list = 3`.

CRCs are parsed but not verified: CRC32C is not in the standard library, and a corrupt record
fails loudly at the protobuf layer anyway. `verify_crc=True` opts in when `crc32c` is installed.

**Key layout.** TFDS flattens an RLDS episode's `steps` sequence into repeated values under
dotted/slashed keys — `steps/action`, `steps/observation/image`,
`steps/language_instruction`. Producers differ in their exact prefixes, so keys are discovered
by suffix rather than matched literally, and `KEY_HINTS` documents what is recognised. A
dataset using different names needs a mapping, not a new reader.
"""

from __future__ import annotations

import struct

# Wire types (protobuf).
_VARINT, _FIXED64, _LENGTH_DELIM, _FIXED32 = 0, 1, 2, 5

# Which field of an RLDS step each recognised key suffix feeds. Matched on the key's tail, so
# "steps/observation/state" and "observation/state" both resolve to the state vector.
KEY_HINTS = {
    "action": ("action",),
    "state": ("observation/state", "observation/robot_state", "state"),
    "image": ("observation/image", "observation/rgb", "image"),
    "instruction": ("language_instruction", "natural_language_instruction", "instruction"),
    "reward": ("reward",),
    "is_terminal": ("is_terminal", "is_last"),
    "confidence": ("action_confidence", "confidence"),
    "force": ("contact_force", "force"),
}


class TFRecordError(ValueError):
    """Raised when the container or the embedded protobuf is not readable."""


# --- protobuf wire format -------------------------------------------------------------------


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf):
            raise TFRecordError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise TFRecordError("varint too long")


def _iter_fields(buf: bytes):
    """Yield `(field_number, wire_type, value)` for one protobuf message."""
    pos = 0
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        field, wire = tag >> 3, tag & 0x07
        if wire == _VARINT:
            value, pos = _read_varint(buf, pos)
        elif wire == _FIXED64:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == _LENGTH_DELIM:
            length, pos = _read_varint(buf, pos)
            value, pos = buf[pos : pos + length], pos + length
        elif wire == _FIXED32:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise TFRecordError(f"unsupported wire type {wire}")
        yield field, wire, value


def _packed_floats(buf: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(buf) // 4}f", buf[: len(buf) // 4 * 4]))


def _packed_varints(buf: bytes) -> list[int]:
    out, pos = [], 0
    while pos < len(buf):
        value, pos = _read_varint(buf, pos)
        out.append(value)
    return out


def _parse_feature(buf: bytes):
    """A `Feature` -> a python list (bytes / float / int), or None if empty."""
    for field, _wire, value in _iter_fields(buf):
        if field == 1:  # bytes_list
            return [v for _f, _w, v in _iter_fields(value)]
        if field == 2:  # float_list
            out: list[float] = []
            for _f, wire, v in _iter_fields(value):
                out.extend(_packed_floats(v) if wire == _LENGTH_DELIM else _packed_floats(v))
            return out
        if field == 3:  # int64_list
            out_i: list[int] = []
            for _f, wire, v in _iter_fields(value):
                out_i.extend(_packed_varints(v) if wire == _LENGTH_DELIM else [v])
            return out_i
    return None


def parse_example(payload: bytes) -> dict[str, list]:
    """A serialized `tf.train.Example` -> `{feature_name: [values]}`."""
    features: dict[str, list] = {}
    for field, _wire, value in _iter_fields(payload):
        if field != 1:  # Example.features
            continue
        for entry_field, _w, entry in _iter_fields(value):
            if entry_field != 1:  # Features.feature (the map)
                continue
            key = None
            parsed = None
            for map_field, _mw, map_value in _iter_fields(entry):
                if map_field == 1:
                    key = map_value.decode("utf-8", "replace")
                elif map_field == 2:
                    parsed = _parse_feature(map_value)
            if key is not None:
                features[key] = parsed if parsed is not None else []
    return features


# --- TFRecord container ---------------------------------------------------------------------


def iter_tfrecords(raw: bytes, verify_crc: bool = False):
    """Yield each record's payload bytes from a TFRecord stream."""
    pos, total = 0, len(raw)
    while pos < total:
        if pos + 12 > total:
            raise TFRecordError("truncated record header")
        (length,) = struct.unpack_from("<Q", raw, pos)
        length_crc = struct.unpack_from("<I", raw, pos + 8)[0]
        pos += 12
        if pos + length + 4 > total:
            raise TFRecordError("record length exceeds the remaining stream")
        payload = raw[pos : pos + length]
        payload_crc = struct.unpack_from("<I", raw, pos + length)[0]
        pos += length + 4

        if verify_crc:
            _verify(raw[pos - length - 16 : pos - length - 8], length_crc)
            _verify(payload, payload_crc)
        yield payload


def _verify(data: bytes, expected: int) -> None:
    try:
        import crc32c
    except ImportError as e:  # pragma: no cover - opt-in path
        raise TFRecordError("verify_crc=True requires the 'crc32c' package") from e
    value = crc32c.crc32c(data)
    masked = (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF
    if masked != expected:
        raise TFRecordError("CRC mismatch — the TFRecord stream is corrupt")


# --- RLDS episode mapping --------------------------------------------------------------------


def _find(features: dict[str, list], kind: str):
    """The value for a recognised key `kind`, matched on key suffix."""
    for suffix in KEY_HINTS[kind]:
        for key, value in features.items():
            normalized = key.replace(".", "/")
            if normalized == suffix or normalized.endswith("/" + suffix):
                return value
    return None


def _chunk(flat: list, n_steps: int) -> list[list]:
    """Split a flattened per-step vector sequence into `n_steps` equal vectors."""
    if not flat or n_steps <= 0 or len(flat) % n_steps:
        return []
    width = len(flat) // n_steps
    return [flat[i * width : (i + 1) * width] for i in range(n_steps)]


def example_to_episode(
    features: dict[str, list],
    *,
    embodiment_type: str = "unknown",
    policy_name: str = "unknown",
    max_frames: int | None = None,
):
    """One parsed `Example` -> a `NormalizedEpisode`, or None if it carries no steps."""
    import base64
    import logging

    from aperture.ingestion.schemas import NormalizedEpisode, NormalizedFrame

    images = _find(features, "image") or []
    rewards = _find(features, "reward") or []
    instructions = _find(features, "instruction") or []
    confidences = _find(features, "confidence") or []
    forces = _find(features, "force") or []
    actions_flat = _find(features, "action") or []
    states_flat = _find(features, "state") or []

    # Step count: whichever per-step feature is present and unambiguous.
    n_steps = len(images) or len(rewards) or len(instructions) or 0
    if not n_steps and actions_flat:
        n_steps = 1
    if not n_steps:
        return None

    if max_frames is not None and n_steps > max_frames:
        logging.getLogger(__name__).warning(
            "RLDS episode truncated from %d to %d steps by max_frames", n_steps, max_frames
        )
        n_steps = max_frames

    actions = _chunk(actions_flat, len(images) or n_steps)
    states = _chunk(states_flat, len(images) or n_steps)

    instruction = None
    if instructions:
        first = instructions[0]
        instruction = first.decode("utf-8", "replace") if isinstance(first, bytes) else str(first)

    frames = [
        NormalizedFrame(
            t=t,
            action=[float(v) for v in actions[t]] if t < len(actions) else None,
            state=[float(v) for v in states[t]] if t < len(states) else None,
            action_confidence=float(confidences[t]) if t < len(confidences) else None,
            contact_force=float(forces[t]) if t < len(forces) else None,
            image_b64=(
                base64.b64encode(images[t]).decode()
                if t < len(images) and isinstance(images[t], bytes)
                else None
            ),
        )
        for t in range(n_steps)
    ]

    final_reward = float(rewards[-1]) if rewards else None
    outcome = "success" if final_reward is not None and final_reward >= 1.0 else "fail"

    return NormalizedEpisode(
        source_format="rlds",
        embodiment_type=embodiment_type,
        policy_name=policy_name,
        instruction=instruction,
        outcome=outcome,
        frames=frames,
        raw_blob=None,
        raw_blob_name=None,
    )


def parse_tfrecord_episodes(raw: bytes, **kwargs) -> list:
    """Every RLDS episode in a `.tfrecord` stream, as `NormalizedEpisode`s."""
    episodes = []
    for payload in iter_tfrecords(raw):
        episode = example_to_episode(parse_example(payload), **kwargs)
        if episode is not None:
            episodes.append(episode)
    if not episodes:
        raise TFRecordError("no RLDS episodes found — recognised no per-step features.")
    return episodes
