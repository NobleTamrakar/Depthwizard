"""PyTorch Dataset for the GAMUS single-view height-estimation benchmark.

GAMUS pairs a nadir aerial RGB chip with a per-pixel AGL (height-above-ground,
meters) map. Layout on disk:

    <root>/images/<split>/<id>_RGB.h5   or  <id>_IMG.h5   -> dataset "image", HxWx3 uint8
    <root>/heights/<split>/<id>_AGL.h5                    -> dataset "image", HxW float32 (meters)

Two source cities are mixed into each split (DC uses "_RGB", NYC uses "_IMG")
so pairing is done by stripping either suffix, not by a fixed one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

_IMAGE_SUFFIX_RE = re.compile(r"_(RGB|IMG)\.h5$")
_HEIGHT_SUFFIX_RE = re.compile(r"_AGL\.h5$")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class GamusPair:
    tile_id: str
    image_path: Path
    height_path: Path


def index_split(root: Path, split: str) -> list[GamusPair]:
    """Pair up RGB/IMG chips with their AGL height maps for one split."""
    image_dir = root / "images" / split
    height_dir = root / "heights" / split

    heights_by_id = {_HEIGHT_SUFFIX_RE.sub("", p.name): p for p in height_dir.glob("*.h5")}

    pairs: list[GamusPair] = []
    for image_path in sorted(image_dir.glob("*.h5")):
        tile_id = _IMAGE_SUFFIX_RE.sub("", image_path.name)
        height_path = heights_by_id.get(tile_id)
        if height_path is None:
            continue  # unpaired file; skip rather than fail the whole split
        pairs.append(GamusPair(tile_id=tile_id, image_path=image_path, height_path=height_path))
    return pairs


class GamusDataset(Dataset):
    """Yields (image_tensor, height_tensor, valid_mask) for DA V2 fine-tuning.

    image_tensor: 3xSxS float32, ImageNet-normalized (matches DA V2's own
        preprocessing in depth_anything_v2/dpt.py:image2tensor).
    height_tensor: SxS float32, AGL height in meters (0 = ground).
    valid_mask: SxS bool, False where height is NaN/negative (sensor holes).
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        input_size: int = 350,
        max_samples: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        assert input_size % 14 == 0, "DA V2's ViT patch size is 14; input_size must be a multiple of it"
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.pairs = index_split(self.root, split)
        if not self.pairs:
            raise FileNotFoundError(f"No GAMUS pairs found under {self.root} for split '{split}'")

        if max_samples is not None and max_samples < len(self.pairs):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.pairs), size=max_samples, replace=False)
            self.pairs = [self.pairs[i] for i in sorted(idx)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]

        with h5py.File(pair.image_path, "r") as f:
            image = f["image"][...]  # HxWx3 uint8
        with h5py.File(pair.height_path, "r") as f:
            height = f["image"][...].astype(np.float32)  # HxW meters

        size = self.input_size
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        height = cv2.resize(height, (size, size), interpolation=cv2.INTER_NEAREST)

        valid = np.isfinite(height) & (height >= 0)
        height = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)

        image_f = image.astype(np.float32) / 255.0
        image_f = (image_f - IMAGENET_MEAN) / IMAGENET_STD
        image_chw = np.transpose(image_f, (2, 0, 1))

        return (
            torch.from_numpy(image_chw.copy()).float(),
            torch.from_numpy(height.copy()).float(),
            torch.from_numpy(valid.copy()),
        )
