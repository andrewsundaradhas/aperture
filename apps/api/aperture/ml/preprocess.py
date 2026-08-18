"""Image preprocessing for the learned policy.

CRITICAL — match the training pipeline exactly. The training notebook fed LeRobot image tensors
(already float RGB in [0, 1], channels-first) through a single `T.Resize((224, 224))` and nothing
else. It did **not** apply ImageNet mean/std normalization. To keep the published checkpoint
valid we replicate that precisely: decode -> RGB float in [0, 1] -> resize to 224x224. Adding the
usual timm/ImageNet normalization here would silently shift the input distribution the weights
were trained on and degrade every prediction.

Imported lazily (only under the `[ml]` extra).
"""

from __future__ import annotations

import io

import torch
import torchvision.transforms as T
from PIL import Image

IMAGE_SIZE = 224

_resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)


def image_bytes_to_tensor(data: bytes, device: str = "cpu") -> torch.Tensor:
    """Decode raw image bytes (PNG/JPEG/…) to a `[1, 3, 224, 224]` float tensor in [0, 1].

    Returns a batch of one so it drops straight into `AperturePolicy.forward`.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    # PILToTensor gives uint8 [0,255]; scale to [0,1] float to match the LeRobot tensors used
    # in training (which were already normalized to [0,1]).
    tensor = T.functional.pil_to_tensor(img).float() / 255.0  # [3, H, W]
    tensor = _resize(tensor)
    return tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]
