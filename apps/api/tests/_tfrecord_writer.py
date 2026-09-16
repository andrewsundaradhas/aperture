"""A tf.train.Example / TFRecord writer for the tests, built straight from the protobuf spec.

Deliberately written independently of `aperture.ingestion.rlds_tfrecord` — it shares no helper
with the reader — so a round-trip test checks the reader against the format rather than against
itself. Encoding here is the inverse of the spec quoted in that module's docstring.
"""
import struct


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def tag(field: int, wire: int) -> bytes:
    return varint((field << 3) | wire)


def ld(field: int, payload: bytes) -> bytes:
    """One length-delimited field."""
    return tag(field, 2) + varint(len(payload)) + payload


def float_list(values) -> bytes:
    packed = struct.pack(f"<{len(values)}f", *values)
    return ld(2, ld(1, packed))          # Feature.float_list -> FloatList.value (packed)


def int64_list(values) -> bytes:
    packed = b"".join(varint(v) for v in values)
    return ld(3, ld(1, packed))          # Feature.int64_list -> Int64List.value (packed)


def bytes_list(values) -> bytes:
    inner = b"".join(ld(1, v) for v in values)
    return ld(1, inner)                  # Feature.bytes_list -> BytesList.value (repeated)


def example(features: dict) -> bytes:
    entries = b"".join(
        ld(1, ld(1, k.encode()) + ld(2, v)) for k, v in features.items()
    )                                     # Features.feature map entries
    return ld(1, entries)                 # Example.features


def tfrecord(payloads) -> bytes:
    """Container framing. CRCs are written as zero — the reader does not verify by default."""
    out = bytearray()
    for p in payloads:
        out += struct.pack("<Q", len(p)) + struct.pack("<I", 0) + p + struct.pack("<I", 0)
    return bytes(out)
