"""
Histopathology Preprocessing Module

Implements standard preprocessing for H&E stained histopathology images:
- Macenko stain normalization (color standardization across labs/scanners)
- Color augmentation (stain-aware jitter)
- Geometric augmentation (rotation/flip — histopath images are rotation-invariant)
- Inter-dataset balancing (equalize class representation across datasets)

Usage:
    from histolab.preprocessing import StainNormalizer, HistoAugmentor, balance_multi_dataset

    # Stain normalization
    normalizer = StainNormalizer(method="macenko")
    normalized_img = normalizer.normalize(pil_image)

    # Augmentation pipeline
    augmentor = HistoAugmentor(stain_jitter=True, geometric=True)
    augmented_img = augmentor(pil_image)

    # Inter-dataset balancing
    balanced_samples = balance_multi_dataset(dataset_splits, target_per_class=100)

References:
    - Macenko et al., "A method for normalizing histology slides for quantitative analysis" (ISBI 2009)
    - Tellez et al., "Quantifying the effects of data augmentation and stain color normalization
      in convolutional neural networks for computational pathology" (MedIA 2019)
"""

import logging
import random
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stain Normalization
# ---------------------------------------------------------------------------

class StainNormalizer:
    """
    Macenko stain normalization for H&E histopathology images.

    Decomposes the image into Hematoxylin and Eosin stain channels using
    SVD on the optical density space, then maps them to a reference
    stain matrix so colors are consistent across different scanners/labs.
    """

    # Reference stain matrix (H&E) — columns are [Hematoxylin, Eosin] in OD space
    # These values are from the original Macenko paper and are widely used.
    REF_STAIN_MATRIX = np.array([
        [0.5626, 0.2159],  # R channel
        [0.7201, 0.8012],  # G channel
        [0.4062, 0.5581],  # B channel
    ])
    REF_MAX_CONC = np.array([1.9705, 1.0308])  # 99th-percentile concentration

    def __init__(self, method: str = "macenko", luminosity_threshold: float = 0.8):
        """
        Args:
            method: Normalization method ("macenko" or "reinhard")
            luminosity_threshold: Pixels brighter than this (in [0,1]) are
                                  treated as background and excluded from fitting.
        """
        if method not in ("macenko", "reinhard"):
            raise ValueError(f"Unknown method: {method}. Use 'macenko' or 'reinhard'.")
        self.method = method
        self.luminosity_threshold = luminosity_threshold

    # -- public API ----------------------------------------------------------

    def normalize(self, image: Image.Image) -> Image.Image:
        """Normalize stain colors of an H&E image."""
        if self.method == "macenko":
            return self._macenko_normalize(image)
        else:
            return self._reinhard_normalize(image)

    # -- Macenko implementation ----------------------------------------------

    def _macenko_normalize(self, image: Image.Image) -> Image.Image:
        img = np.array(image.convert("RGB")).astype(np.float64)
        h, w, _ = img.shape

        # 1. Convert to optical density
        img_od = self._rgb_to_od(img)

        # 2. Mask out background (low OD = bright pixels)
        od_flat = img_od.reshape(-1, 3)
        mask = np.any(od_flat > 0.15, axis=1)
        tissue_od = od_flat[mask]

        if tissue_od.shape[0] < 10:
            # Not enough tissue — return original
            return image

        # 3. SVD to find stain vectors
        try:
            stain_matrix, max_conc = self._get_stain_matrix(tissue_od)
        except Exception:
            return image

        # 4. Get concentrations of each pixel
        conc = self._get_concentrations(od_flat, stain_matrix)

        # 5. Scale concentrations to reference
        max_conc_clipped = np.clip(max_conc, 1e-6, None)
        conc_normalized = conc * (self.REF_MAX_CONC / max_conc_clipped)

        # 6. Reconstruct in reference stain space
        od_normalized = conc_normalized @ self.REF_STAIN_MATRIX.T
        rgb_normalized = self._od_to_rgb(od_normalized)
        rgb_normalized = rgb_normalized.reshape(h, w, 3)
        rgb_normalized = np.clip(rgb_normalized, 0, 255).astype(np.uint8)

        return Image.fromarray(rgb_normalized)

    def _get_stain_matrix(self, tissue_od: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 2x3 stain matrix via SVD + angular extremes."""
        # Center and SVD
        _, _, Vt = np.linalg.svd(tissue_od - tissue_od.mean(axis=0), full_matrices=False)
        # Project onto first two principal directions
        plane = Vt[:2, :]  # (2, 3)
        proj = tissue_od @ plane.T  # (N, 2)

        # Find angular extremes
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        min_angle = np.percentile(angles, 1)
        max_angle = np.percentile(angles, 99)

        v1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ plane
        v2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ plane

        # Ensure positive and normalize
        v1 = np.abs(v1)
        v2 = np.abs(v2)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        # Convention: Hematoxylin is the vector with larger first component (more blue)
        if v1[0] < v2[0]:
            stain_matrix = np.column_stack([v1, v2])
        else:
            stain_matrix = np.column_stack([v2, v1])

        # Get max concentrations (99th percentile)
        conc = self._get_concentrations(tissue_od, stain_matrix)
        max_conc = np.percentile(conc, 99, axis=0)

        return stain_matrix, max_conc

    @staticmethod
    def _get_concentrations(od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Solve for stain concentrations via least-squares."""
        # od = conc @ stain_matrix.T  =>  conc = od @ pinv(stain_matrix.T)
        return od @ np.linalg.pinv(stain_matrix.T)

    @staticmethod
    def _rgb_to_od(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB [0-255] to optical density."""
        rgb = np.clip(rgb, 1, 255).astype(np.float64)
        return -np.log(rgb / 255.0)

    @staticmethod
    def _od_to_rgb(od: np.ndarray) -> np.ndarray:
        """Convert optical density back to RGB [0-255]."""
        return 255.0 * np.exp(-od)

    # -- Reinhard implementation ---------------------------------------------

    # Reference mean/std in LAB space (from a "typical" H&E slide)
    _REF_LAB_MEAN = np.array([70.0, 3.0, -12.0])
    _REF_LAB_STD = np.array([15.0, 8.0, 8.0])

    def _reinhard_normalize(self, image: Image.Image) -> Image.Image:
        """Simple Reinhard color transfer in LAB space."""
        from PIL import ImageCms

        img = np.array(image.convert("RGB")).astype(np.float64)
        # Convert to LAB (approximate via linear transform)
        lab = self._rgb_to_lab(img)

        # Compute source stats on tissue pixels only
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        mask = gray < (self.luminosity_threshold * 255)

        if mask.sum() < 10:
            return image

        for c in range(3):
            channel = lab[:, :, c]
            src_mean = channel[mask].mean()
            src_std = channel[mask].std() + 1e-6
            lab[:, :, c] = (channel - src_mean) * (self._REF_LAB_STD[c] / src_std) + self._REF_LAB_MEAN[c]

        rgb = self._lab_to_rgb(lab)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb)

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        """Approximate RGB->LAB conversion."""
        # Normalize to [0,1]
        rgb_norm = rgb / 255.0
        # Linearize (sRGB gamma)
        linear = np.where(rgb_norm > 0.04045,
                          ((rgb_norm + 0.055) / 1.055) ** 2.4,
                          rgb_norm / 12.92)
        # To XYZ (D65)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ])
        xyz = linear @ M.T
        # Normalize by D65 white point
        xyz /= np.array([0.95047, 1.0, 1.08883])

        # To LAB
        def f(t):
            return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)

        fx, fy, fz = f(xyz[:, :, 0]), f(xyz[:, :, 1]), f(xyz[:, :, 2])
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return np.stack([L, a, b], axis=-1)

    @staticmethod
    def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
        """Approximate LAB->RGB conversion."""
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        def f_inv(t):
            return np.where(t > 0.206893, t ** 3, (t - 16 / 116) / 7.787)

        xyz = np.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=-1)
        xyz *= np.array([0.95047, 1.0, 1.08883])

        M_inv = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ])
        linear = xyz @ M_inv.T
        linear = np.clip(linear, 0, 1)

        # Gamma
        rgb = np.where(linear > 0.0031308,
                       1.055 * (linear ** (1 / 2.4)) - 0.055,
                       12.92 * linear)
        return rgb * 255


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

class HistoAugmentor:
    """
    Augmentation pipeline tailored for histopathology images.

    Histopath patches are rotation-invariant (no "up"), so aggressive
    geometric transforms are appropriate. Color jitter simulates
    inter-lab stain variation.
    """

    def __init__(
        self,
        geometric: bool = True,
        color_jitter: bool = True,
        stain_jitter: bool = False,
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        saturation_range: Tuple[float, float] = (0.85, 1.15),
        hue_shift_range: int = 10,
    ):
        """
        Args:
            geometric: Enable random 90-degree rotation + horizontal/vertical flip
            color_jitter: Enable brightness/contrast/saturation jitter
            stain_jitter: Enable stain-channel-level color perturbation (slower)
            brightness_range: (min, max) multiplier for brightness
            contrast_range: (min, max) multiplier for contrast
            saturation_range: (min, max) multiplier for saturation
            hue_shift_range: Max hue shift in degrees (applied in HSV space)
        """
        self.geometric = geometric
        self.color_jitter = color_jitter
        self.stain_jitter = stain_jitter
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_shift_range = hue_shift_range

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply augmentation pipeline to an image."""
        img = image.copy()

        if self.geometric:
            img = self._geometric_augment(img)

        if self.color_jitter:
            img = self._color_jitter(img)

        if self.stain_jitter:
            img = self._stain_jitter(img)

        return img

    @staticmethod
    def _geometric_augment(img: Image.Image) -> Image.Image:
        """Random 90-degree rotation + flip."""
        # Random rotation: 0, 90, 180, 270
        k = random.randint(0, 3)
        if k > 0:
            img = img.rotate(k * 90, expand=False)

        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img

    def _color_jitter(self, img: Image.Image) -> Image.Image:
        """Random brightness, contrast, saturation adjustments."""
        # Brightness
        factor = random.uniform(*self.brightness_range)
        img = ImageEnhance.Brightness(img).enhance(factor)

        # Contrast
        factor = random.uniform(*self.contrast_range)
        img = ImageEnhance.Contrast(img).enhance(factor)

        # Saturation
        factor = random.uniform(*self.saturation_range)
        img = ImageEnhance.Color(img).enhance(factor)

        # Hue shift (in numpy HSV)
        if self.hue_shift_range > 0:
            arr = np.array(img)
            # Simple approximate hue shift by channel rotation
            shift = random.randint(-self.hue_shift_range, self.hue_shift_range)
            if shift != 0:
                hsv = self._rgb_to_hsv(arr)
                hsv[:, :, 0] = (hsv[:, :, 0] + shift / 360.0) % 1.0
                arr = self._hsv_to_rgb(hsv)
                img = Image.fromarray(arr)

        return img

    def _stain_jitter(self, img: Image.Image) -> Image.Image:
        """
        Perturbation in stain space — adds random noise to H and E
        concentration channels independently, producing realistic
        inter-lab stain variation.
        """
        arr = np.array(img).astype(np.float64)
        od = StainNormalizer._rgb_to_od(arr)
        od_flat = od.reshape(-1, 3)

        # Quick approximate stain separation using reference matrix
        conc = od_flat @ np.linalg.pinv(StainNormalizer.REF_STAIN_MATRIX.T)

        # Perturb each stain channel independently
        for i in range(2):
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-0.05, 0.05)
            conc[:, i] = conc[:, i] * alpha + beta

        # Reconstruct
        od_new = conc @ StainNormalizer.REF_STAIN_MATRIX.T
        rgb_new = StainNormalizer._od_to_rgb(od_new).reshape(arr.shape)
        rgb_new = np.clip(rgb_new, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb_new)

    @staticmethod
    def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """RGB [0-255] -> HSV [0-1, 0-1, 0-255]."""
        rgb_f = rgb.astype(np.float64) / 255.0
        maxc = rgb_f.max(axis=2)
        minc = rgb_f.min(axis=2)
        diff = maxc - minc + 1e-10

        h = np.zeros_like(maxc)
        r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]

        mask_r = maxc == r
        mask_g = (~mask_r) & (maxc == g)
        mask_b = (~mask_r) & (~mask_g)

        h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
        h[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
        h[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
        h /= 6.0

        s = np.where(maxc > 0, diff / (maxc + 1e-10), 0)
        v = maxc

        return np.stack([h, s, v], axis=-1)

    @staticmethod
    def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
        """HSV [0-1, 0-1, 0-1] -> RGB [0-255]."""
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        i = (h * 6).astype(int) % 6
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        rgb = np.zeros_like(hsv)
        for idx, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                          (p, q, v), (t, p, v), (v, p, q)]):
            mask = i == idx
            rgb[mask, 0] = r[mask]
            rgb[mask, 1] = g[mask]
            rgb[mask, 2] = b[mask]

        return (rgb * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inter-Dataset Balancing
# ---------------------------------------------------------------------------

def balance_multi_dataset(
    dataset_splits: Dict[str, "DatasetSplit"],
    target_per_class: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Balance training samples across multiple datasets so each CLASS
    (not each dataset) contributes equally.

    Without this, when combining CRC (9 classes, ~111 per class from 1000
    total) with PCam (2 classes, ~500 per class), PCam classes dominate
    the training signal 5:1.

    Strategy: compute `target_per_class` as the median per-class count
    across all datasets, then oversample small classes and undersample
    large ones to equalize.

    Args:
        dataset_splits: Dict mapping dataset name to DatasetSplit
                        (from MultiDatasetLoader.load_basket)
        target_per_class: Target samples per class. If None, uses median.

    Returns:
        (balanced_train, balanced_val) — flat lists of sample dicts
    """
    # Count samples per class across all datasets
    class_samples_train = defaultdict(list)
    class_samples_val = defaultdict(list)

    for ds_name, split in dataset_splits.items():
        for sample in split.train:
            label = sample.get("label_name", sample.get("label", "unknown"))
            key = f"{ds_name}/{label}"
            class_samples_train[key].append(sample)

        for sample in split.validation:
            label = sample.get("label_name", sample.get("label", "unknown"))
            key = f"{ds_name}/{label}"
            class_samples_val[key].append(sample)

    # Determine target
    counts = [len(v) for v in class_samples_train.values()]
    if not counts:
        return [], []

    if target_per_class is None:
        target_per_class = int(np.median(counts))
    target_per_class = max(target_per_class, 1)

    logger.info(
        f"Inter-dataset balancing: {len(class_samples_train)} classes, "
        f"range [{min(counts)}, {max(counts)}], target={target_per_class}"
    )

    # Resample train set
    balanced_train = []
    for key, samples in class_samples_train.items():
        n = len(samples)
        if n >= target_per_class:
            # Undersample
            balanced_train.extend(random.sample(samples, target_per_class))
        else:
            # Oversample (with replacement)
            balanced_train.extend(samples)
            extra = target_per_class - n
            balanced_train.extend(random.choices(samples, k=extra))
        logger.info(f"  {key}: {n} -> {target_per_class}")

    # Val set: keep original (no resampling to preserve real distribution)
    balanced_val = []
    for samples in class_samples_val.values():
        balanced_val.extend(samples)

    random.shuffle(balanced_train)
    random.shuffle(balanced_val)

    logger.info(f"Balanced train: {len(balanced_train)}, val: {len(balanced_val)}")
    return balanced_train, balanced_val
