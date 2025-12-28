import os, json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transform import CourtTransform


class CocoBallDataset(Dataset):
    """
    Dataset for object detection with a single class: 'ball'.
    Returns image + bounding boxes + labels.
    Compatible with RT-DETR, DETR, YOLO-like models.
    """

    def __init__(self, ann_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # id -> info
        self.images = {img["id"]: img for img in coco["images"]}

        df = pd.DataFrame(coco["annotations"])
        self.ann_by_image = (
            df.groupby("image_id")
              .apply(lambda x: x.to_dict("records"))
              .to_dict()
        )

        self.image_ids = list(self.ann_by_image.keys())
        print(f"[CocoBallDataset] Loaded {len(self.image_ids)} samples")

        # only 1 class
        self.class_name_to_id = {"ball": 0}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        anns = self.ann_by_image[image_id]
        img_info = self.images[image_id]

        # load RGB image (uint8 - fastest for ffmpeg)
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.uint8)   # (H,W,3)

        # collect boxes
        bboxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])  # xyxy format
            labels.append(0)  # only class "ball"

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # apply transform (if exists)
        if self.transform:
            img, bboxes = self.transform(img, bboxes)

        img = torch.tensor(img, dtype=torch.float32)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return (
            img,                 # (3,H,W)
            bboxes,              # (N,4)
            labels,              # (N,)
            # "image_id": img_info["file_name"].split(".")[0]
        )

class CocoCourtDataset(Dataset):
    def __init__(self, ann_file, img_dir, target_size=(640, 640), scale=2, hp_radius=7, train=True):
        self.img_dir = img_dir
        self.scale = scale
        self.hp_radius = hp_radius
        self.output_h = target_size[0] // scale
        self.output_w = target_size[1] // scale
        self.transform = CourtTransform(target_size=target_size, train=train)

        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = coco["categories"]
        self.num_keypoints = len(self.categories[1]["keypoints"])
        df = pd.DataFrame(coco["annotations"])
        self.ann_by_image = df.groupby("image_id").apply(lambda x: x.to_dict("records")).to_dict()
        self.image_ids = list(self.ann_by_image.keys())

        print(f"[CourtDataset] Loaded {len(self.image_ids)} images | {self.num_keypoints} keypoints")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        anns = self.ann_by_image[image_id]
        img_info = self.images[image_id]

        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        # NEW! Astype to np.uint8
        img = np.array(img).astype(np.uint8)

        ann = max(anns, key=lambda a: a.get("area", 0))
        kps = np.array(ann["keypoints"]).reshape(-1, 3)
        kps = torch.tensor(kps[:, :2], dtype=torch.float32)

        img, kps = self.transform(img, kps)

        # hm_hp = np.zeros((self.num_keypoints, self.output_h, self.output_w), dtype=np.float32)

        return (
            img,
            # torch.tensor(hm_hp, dtype=torch.float32),
            kps.to(torch.int32),
            img_info["file_name"].split(".")[0],
        )
