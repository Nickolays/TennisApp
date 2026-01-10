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
            train: enable augmentation if True
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

        # --- AUGMENTATIONS (training only) ---
        # NO AUGMENTATION - removed to eliminate train/val gap
        # Bug fix: validation was being augmented because we used random_split
        # on a single dataset with train=True
        pass

        # --- Resize image ---
        img = cv2.resize(img, (self.out_w, self.out_h))
        h_before, w_before = img.shape[:2] if keypoints is None else (H - 2*dy, W - 2*dx)
        scale_x = self.out_w / w_before
        scale_y = self.out_h / h_before
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

    def _rotate(self, img, keypoints, angle):
        """Rotate image and keypoints around center"""
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if keypoints is not None:
            # Transform keypoints: [x, y, 1] @ M.T
            ones = np.ones((keypoints.shape[0], 1))
            kps_homogeneous = np.hstack([keypoints, ones])
            keypoints = (M @ kps_homogeneous.T).T
        return img, keypoints

    def _perspective_transform(self, img, keypoints, margin_pct=0.05):
        """Apply random perspective warp (simulate camera angles)"""
        h, w = img.shape[:2]
        margin = int(min(h, w) * margin_pct)

        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if keypoints is not None:
            # Transform keypoints to homogeneous coordinates
            ones = np.ones((keypoints.shape[0], 1))
            kps_h = np.hstack([keypoints, ones])
            kps_transformed = (M @ kps_h.T).T
            # Convert back from homogeneous
            keypoints = kps_transformed[:, :2] / kps_transformed[:, 2:]
        return img, keypoints

    def _color_jitter(self, img):
        """Random hue/saturation adjustments in HSV space (REDUCED)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] += random.uniform(-5, 5)        # hue shift (was -8 to 8)
        hsv[..., 1] *= random.uniform(0.9, 1.1)     # saturation (was 0.85-1.15)
        hsv[..., 2] *= random.uniform(0.9, 1.1)     # value (was 0.85-1.15)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)