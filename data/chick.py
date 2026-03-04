# data/chick.py
import os
import random
from glob import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DatasetChick(Dataset):

    def __init__(self, datapath, fold, transform, split, shot=1):

        self.root = os.path.join(datapath, "chick")

        self.transform = transform
        self.split = split
        self.shot = int(shot)

        self.benchmark = "chick"
        self.nclass = 1
        self.class_ids = [0]

        self.train_img_dir = os.path.join(self.root, "train/images")
        self.train_msk_dir = os.path.join(self.root, "train/segmentations")

        self.test_img_dir = os.path.join(self.root, "test/images")
        self.test_msk_dir = os.path.join(self.root, "test/segmentations")

        self.train_ids = self._collect_ids(self.train_img_dir, self.train_msk_dir)
        self.test_ids = self._collect_ids(self.test_img_dir, self.test_msk_dir)

        if split == "trn":
            self.episodes = self.train_ids
        else:
            self.episodes = self.test_ids

    def __len__(self):
        return len(self.episodes)

    def _collect_ids(self, img_dir, mask_dir):

        imgs = glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png"))

        ids = []

        for p in imgs:

            stem = os.path.splitext(os.path.basename(p))[0]

            mask_path = os.path.join(mask_dir, stem + ".png")

            if os.path.exists(mask_path):
                ids.append(stem)

        return sorted(ids)

    def _load_img(self, img_dir, stem):

        path_jpg = os.path.join(img_dir, stem + ".jpg")
        path_png = os.path.join(img_dir, stem + ".png")

        path = path_jpg if os.path.exists(path_jpg) else path_png

        return Image.open(path).convert("RGB")

    def _load_mask(self, mask_dir, stem):

        path = os.path.join(mask_dir, stem + ".png")

        m = Image.open(path).convert("L")

        m = np.array(m)

        m = (m > 0).astype(np.uint8)

        return m

    def _resize_mask(self, mask, size):

        return np.array(
            Image.fromarray(mask).resize(size, resample=Image.NEAREST)
        )

    def __getitem__(self, idx):

        q_stem = self.episodes[idx]

        if self.split == "trn":

            q_img = self._load_img(self.train_img_dir, q_stem)
            q_mask = self._load_mask(self.train_msk_dir, q_stem)

            pool = [s for s in self.train_ids if s != q_stem]

            if len(pool) == 0:
                pool = self.train_ids

            s_stems = random.sample(pool, min(self.shot, len(pool)))

            s_imgs = [self._load_img(self.train_img_dir, s) for s in s_stems]
            s_masks = [self._load_mask(self.train_msk_dir, s) for s in s_stems]

        else:

            q_img = self._load_img(self.test_img_dir, q_stem)
            q_mask = self._load_mask(self.test_msk_dir, q_stem)

            pool = self.train_ids

            s_stems = random.sample(pool, min(self.shot, len(pool)))

            s_imgs = [self._load_img(self.train_img_dir, s) for s in s_stems]
            s_masks = [self._load_mask(self.train_msk_dir, s) for s in s_stems]

        q_img_t = self.transform(q_img)

        s_imgs_t = torch.stack([self.transform(x) for x in s_imgs], dim=0)

        _, H, W = q_img_t.shape

        q_mask = self._resize_mask(q_mask, (W, H))
        q_mask = torch.from_numpy(q_mask).long()

        s_masks_resized = []

        for m in s_masks:
            m = self._resize_mask(m, (W, H))
            s_masks_resized.append(torch.from_numpy(m).long())

        s_masks_t = torch.stack(s_masks_resized)

        return {
            "query_img": q_img_t,
            "query_mask": q_mask,
            "support_imgs": s_imgs_t,
            "support_masks": s_masks_t,
            "class_id": torch.tensor([0], dtype=torch.long),
        }