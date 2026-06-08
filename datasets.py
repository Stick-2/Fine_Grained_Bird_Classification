from pathlib import Path

import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

import config


def _tfm(train, *, train_crop_padding=None, color_jitter=None):
    s = config.IMAGE_SIZE
    pad = config.TRAIN_CROP_PADDING if train_crop_padding is None else train_crop_padding
    cj = tuple(config.COLOR_JITTER) if color_jitter is None else tuple(color_jitter)
    n = T.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    if train:
        return T.Compose([
            T.Resize((s + pad,) * 2),
            T.RandomCrop(s),
            T.RandomHorizontalFlip(),
            T.ColorJitter(*cj),
            T.ToTensor(),
            n,
        ])
    return T.Compose([T.Resize((s, s)), T.ToTensor(), n])


class CUBDataset(Dataset):
    def __init__(self, root, labels_csv, split, train, *, train_crop_padding=None, color_jitter=None):
        self.root = Path(root)
        preferred = self.root / config.IMAGES_DIR_NAME
        fallback = self.root / config.IMAGES_FALLBACK_DIR_NAME
        self.img_dir = preferred if preferred.exists() else fallback
        self.tfm = _tfm(train, train_crop_padding=train_crop_padding, color_jitter=color_jitter)
        df = pd.read_csv(labels_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = Image.open(self.img_dir / r["image"]).convert("RGB")
        return self.tfm(x), int(r["label"])

    @property
    def num_classes(self):
        return int(self.df["label"].max()) + 1


def get_dataloaders(
    data_root,
    labels_csv,
    *,
    batch_size=None,
    num_workers=None,
    train_crop_padding=None,
    color_jitter=None,
):
    tr = CUBDataset(
        data_root,
        labels_csv,
        "train",
        True,
        train_crop_padding=train_crop_padding,
        color_jitter=color_jitter,
    )
    te = CUBDataset(data_root, labels_csv, "test", False)
    bs = config.BATCH_SIZE if batch_size is None else batch_size
    nw = config.NUM_WORKERS if num_workers is None else num_workers
    kw = dict(batch_size=bs, num_workers=nw)
    a = torch.utils.data.DataLoader(tr, shuffle=True, pin_memory=True, **kw)
    b = torch.utils.data.DataLoader(te, shuffle=False, **kw)
    return a, b, tr.num_classes
