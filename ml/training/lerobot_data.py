"""LeRobot v3.0 dataset reader — re-exported from the API package.

The reader itself lives in `aperture.ingestion.lerobot_v3` so that the ingestion API and this
training code read a dataset through exactly one implementation. Two readers would be two
chances to disagree about frame/row alignment, and a disagreement there is invisible: training
would learn against one pairing of image and action, and the deployed system would attribute
failures against another.

This module stays as the import path `ml/training` and the notebooks already use.
"""

from __future__ import annotations

from aperture.ingestion.lerobot_v3 import (
    HF_DATASET_REPO,
    LeRobotV3Dataset,
    download_dataset,
    pad_actions,
)

__all__ = ["HF_DATASET_REPO", "LeRobotV3Dataset", "download_dataset", "pad_actions"]
