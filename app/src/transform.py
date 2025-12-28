import cv2, torch, random
import numpy as np


class DetectionTransform:
    def __init__(self, target_size=(640, 640), train=True):
        self.h, self.w = target_size
        self.train = train

    def __call__(self, img, bboxes):
        H, W = img.shape[:2]

        # resize (with scale correction)
        img_resized = cv2.resize(img, (self.w, self.h))

        scale_x = self.w / W
        scale_y = self.h / H

        bboxes = bboxes.copy()
        bboxes[:, [0,2]] *= scale_x
        bboxes[:, [1,3]] *= scale_y

        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = np.transpose(img_resized, (3,0,1)) if img_resized.ndim==4 else np.transpose(img_resized, (2,0,1))

        return img_resized, bboxes


class CourtTransform:
    def __init__(self, target_size=(640, 640), crop_ratio=0.1, train=True):
        """
        Args:
            target_size: (H, W) final image size
            crop_ratio: fraction to crop from each border (default 0.1)
            train: enable random horizontal flip if True
        """
        self.out_h, self.out_w = target_size
        self.crop_ratio = crop_ratio
        self.train = train

    def __call__(self, img, keypoints=None):
        """
        Args:
            img: np.ndarray of shape (H, W, C), BGR or RGB
            keypoints: np.ndarray or torch.Tensor of shape (N, 2)
        Returns:
            img_out: np.ndarray (C, H, W) float32 [0,1]
            keypoints_out: torch.Tensor (N, 2)
        """
        assert isinstance(img, np.ndarray), "Expected numpy image input"
        H, W = img.shape[:2]
        dx, dy = int(W * self.crop_ratio), int(H * self.crop_ratio)

        # --- Crop borders ---
        img = img[dy:H - dy, dx:W - dx]
        if keypoints is not None:
            if isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.numpy()
            keypoints = keypoints - np.array([dx, dy])

        # --- Random horizontal flip ---
        if self.train and random.random() < 0.5:
            img = cv2.flip(img, 1)
            if keypoints is not None:
                keypoints[:, 0] = (W - 2 * dx) - keypoints[:, 0]

        # --- Resize image ---
        img = cv2.resize(img, (self.out_w, self.out_h))
        scale_x = self.out_w / (W - 2 * dx)
        scale_y = self.out_h / (H - 2 * dy)
        if keypoints is not None:
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y

        # --- Normalize + convert to tensor-like (C, H, W) ---
        img = img.astype(np.float32) / 255.0
        # ensure RGB order (if input is BGR from cv2)
        if img.shape[2] == 3:
            img = img[..., ::-1]
        img = np.transpose(img, (2, 0, 1))  # (C, H, W)

        keypoints_out = torch.from_numpy(keypoints).float() if keypoints is not None else None
        return img, keypoints_out