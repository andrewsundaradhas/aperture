"""Glue between the API layers and the learned-model runtime.

Keeps the "should we use the learned path?" decision and the frame-image loading in one place,
so the routers stay readable. Imports storage + DB models but never torch — the heavy imports
stay inside `aperture.ml.runtime`, gated by `learned_enabled()`.
"""

from __future__ import annotations

from aperture.core.config import get_settings
from aperture.core.models import Episode
from aperture.core.storage import read_uri
from aperture.ml import runtime


def learned_enabled() -> bool:
    """True iff the operator opted in AND the models can actually run."""
    return bool(get_settings().use_learned_models) and runtime.is_available()


def first_frame_image(episode: Episode) -> bytes | None:
    """Bytes of the earliest frame that carries an image, or None. Frames are ordered by t."""
    for f in episode.frames:
        if f.image_uri:
            try:
                return read_uri(f.image_uri)
            except Exception:
                return None
    return None


def episode_frame_images(episode: Episode, max_frames: int = 8) -> list[tuple[int, bytes]]:
    """(t, image_bytes) for up to `max_frames` frames that have images, in timestep order."""
    out: list[tuple[int, bytes]] = []
    for f in episode.frames:
        if not f.image_uri:
            continue
        try:
            out.append((f.t, read_uri(f.image_uri)))
        except Exception:
            continue
        if len(out) >= max_frames:
            break
    return out


def episode_has_images(episode: Episode) -> bool:
    return any(f.image_uri for f in episode.frames)
