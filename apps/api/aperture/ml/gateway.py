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


def _decodable(data: bytes) -> bool:
    """True if the bytes are an image the model's preprocessing can open.

    Uploads are untrusted: a frame whose image field holds garbage used to reach PIL inside the
    model call and turn `classify` into a 500. A bad image now means "no image", and the episode
    takes the heuristic path like any other image-less episode.
    """
    import io

    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return True
    except Exception:
        return False


def first_frame_image(episode: Episode) -> bytes | None:
    """Bytes of the earliest frame with a readable image, or None. Frames are ordered by t."""
    for f in episode.frames:
        if f.image_uri:
            try:
                data = read_uri(f.image_uri)
            except Exception:
                return None
            return data if _decodable(data) else None
    return None


def episode_frame_images(episode: Episode, max_frames: int = 8) -> list[tuple[int, bytes]]:
    """(t, image_bytes) for up to `max_frames` frames that have images, in timestep order."""
    out: list[tuple[int, bytes]] = []
    for f in episode.frames:
        if not f.image_uri:
            continue
        try:
            data = read_uri(f.image_uri)
        except Exception:
            continue
        if _decodable(data):
            out.append((f.t, data))
        if len(out) >= max_frames:
            break
    return out


def episode_has_images(episode: Episode) -> bool:
    return any(f.image_uri for f in episode.frames)
