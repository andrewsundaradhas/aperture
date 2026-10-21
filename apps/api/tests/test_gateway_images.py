"""A frame whose image bytes are not an image must mean "no image", not a 500."""

from __future__ import annotations

import io

import pytest

PIL = pytest.importorskip("PIL")


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 30)).save(buf, format="PNG")
    return buf.getvalue()


class _Frame:
    def __init__(self, t, image_uri):
        self.t, self.image_uri = t, image_uri


class _Episode:
    def __init__(self, frames):
        self.frames = frames


def test_garbage_image_bytes_are_treated_as_missing(monkeypatch):
    from aperture.ml import gateway

    blobs = {"local://a/bad": b"not an image at all", "local://a/good": _png()}
    monkeypatch.setattr(gateway, "read_uri", lambda uri: blobs[uri])

    assert gateway.first_frame_image(_Episode([_Frame(0, "local://a/bad")])) is None
    assert gateway.first_frame_image(_Episode([_Frame(0, "local://a/good")])) == blobs["local://a/good"]
    got = gateway.episode_frame_images(_Episode([_Frame(0, "local://a/bad"), _Frame(1, "local://a/good")]))
    assert [t for t, _ in got] == [1]
