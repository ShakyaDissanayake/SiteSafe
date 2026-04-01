"""Construction-site-specific image augmentation pipeline.

Designed to address the unique challenges of construction site imagery:
  - Harsh and variable lighting (direct sun, shadows, overcast, artificial)
  - Motion blur from handheld cameras or vibration
  - Partial occlusion from equipment, scaffolding, and other workers
  - Wide range of camera angles (ground level, elevated, surveillance)
  - Hi-vis vest color consistency across lighting conditions

Uses Albumentations for high-performance, composable augmentations.

Usage:
    from augmentations import get_train_augmentations, get_val_augmentations
    transform = get_train_augmentations(image_size=640)
    augmented = transform(image=image, bboxes=bboxes, class_labels=labels)
"""

from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentations(
    image_size: int = 640,
    brightness_limit: float = 0.3,
    contrast_limit: float = 0.3,
    blur_limit: int = 7,
    occlusion_patches: int = 3,
    occlusion_ratio: tuple[float, float] = (0.02, 0.08),
    perspective_scale: float = 0.08,
    hsv_hue_shift: int = 15,
    hsv_sat_shift: int = 40,
    hsv_val_shift: int = 40,
) -> A.Compose:
    """Build the training augmentation pipeline.

    Each augmentation targets a specific construction-site challenge:

    Args:
        image_size: Target image size (square) for model input.
        brightness_limit: Max brightness adjustment fraction.
        contrast_limit: Max contrast adjustment fraction.
        blur_limit: Max kernel size for motion blur.
        occlusion_patches: Max number of random occlusion rectangles.
        occlusion_ratio: Min/max size of occlusion patches (fraction of image).
        perspective_scale: Max perspective warp magnitude.
        hsv_hue_shift: Max hue shift in HSV space.
        hsv_sat_shift: Max saturation shift in HSV space.
        hsv_val_shift: Max value (brightness) shift in HSV space.

    Returns:
        Albumentations Compose pipeline with bbox-safe transforms.
    """
    return A.Compose(
        [
            # ── Spatial transforms ────────────────────────────────────
            # Random resize to handle variable worker distances (close/mid/far)
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.5, 1.0),
                ratio=(0.8, 1.2),
                p=0.5,
            ),
            # Resize to model input (always applied)
            A.Resize(height=image_size, width=image_size),
            # Horizontal flip — construction sites have no left/right bias
            A.HorizontalFlip(p=0.5),
            # Perspective warp — simulates different camera mounting angles
            # (ground-level vs. elevated surveillance cameras)
            A.Perspective(scale=(0.02, perspective_scale), p=0.3),
            # Affine — slight rotation and scale for camera tilt variation
            A.Affine(
                rotate=(-10, 10),
                scale=(0.9, 1.1),
                shear=(-5, 5),
                p=0.3,
            ),

            # ── Color/lighting transforms ────────────────────────────
            # Brightness & contrast — handles harsh sun, deep shadows,
            # overcast conditions common on outdoor construction sites
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.6,
            ),
            # HSV shift — critical for hi-vis vest color robustness.
            # Fluorescent yellows/oranges shift under different lighting;
            # this teaches the model to recognize vests across conditions
            A.HueSaturationValue(
                hue_shift_limit=hsv_hue_shift,
                sat_shift_limit=hsv_sat_shift,
                val_shift_limit=hsv_val_shift,
                p=0.5,
            ),
            # CLAHE — simulates adaptive histogram equalization used
            # as preprocessing for low-light/night scenes
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
            # Random gamma — additional lighting variation
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            # Channel shuffle — rare but helps model not overfit to
            # specific color channel patterns
            A.ChannelShuffle(p=0.05),

            # ── Blur / noise — simulates real capture conditions ─────
            # Motion blur — workers and camera both move; construction
            # sites have vibration from heavy machinery
            A.MotionBlur(blur_limit=blur_limit, p=0.25),
            # Gaussian noise — sensor noise in low-light surveillance cameras
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            # Defocus blur — out-of-focus regions common in auto-focus
            # surveillance cameras tracking moving workers
            A.Defocus(radius=(3, 6), p=0.15),

            # ── Occlusion simulation ─────────────────────────────────
            # CoarseDropout — simulates partial occlusion from scaffold
            # poles, equipment, and other construction elements blocking
            # view of workers and their PPE
            A.CoarseDropout(
                num_holes_range=(1, occlusion_patches),
                hole_height_range=(
                    int(image_size * occlusion_ratio[0]),
                    int(image_size * occlusion_ratio[1]),
                ),
                hole_width_range=(
                    int(image_size * occlusion_ratio[0]),
                    int(image_size * occlusion_ratio[1]),
                ),
                fill="random",
                p=0.3,
            ),

            # ── Weather simulation ───────────────────────────────────
            # Random fog — common on early morning construction sites
            A.RandomFog(
                fog_coef_range=(0.1, 0.3),
                alpha_coef=0.1,
                p=0.1,
            ),
            # Random rain — outdoor construction continues in light rain
            A.RandomRain(
                slant_range=(-10, 10),
                drop_length=15,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.8,
                p=0.1,
            ),
            # Random sun flare — direct sunlight causing lens flare
            A.RandomSunFlare(
                flare_roi=(0, 0, 1.0, 0.5),
                angle_range=(0, 1),
                num_flare_circles_range=(3, 6),
                src_radius=150,
                p=0.1,
            ),

            # ── Normalize (ImageNet stats) ───────────────────────────
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_area=100,           # Drop tiny fragments
            min_visibility=0.3,     # Keep bbox if >30% visible after crop
        ),
    )


def get_val_augmentations(image_size: int = 640) -> A.Compose:
    """Build the validation/test augmentation pipeline (deterministic).

    Only resizing and normalization — no random transforms.

    Args:
        image_size: Target image size (square) for model input.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
        ),
    )


def get_tta_augmentations(image_size: int = 640) -> list[A.Compose]:
    """Build test-time augmentation (TTA) variants.

    Returns multiple transform pipelines to ensemble predictions
    for higher-confidence safety assessments.

    Args:
        image_size: Target image size.

    Returns:
        List of augmentation pipelines for TTA.
    """
    base = A.Resize(height=image_size, width=image_size)
    norm = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    bbox_params = A.BboxParams(
        format="yolo", label_fields=["class_labels"]
    )

    return [
        # Original
        A.Compose([base, norm], bbox_params=bbox_params),
        # Horizontal flip
        A.Compose(
            [base, A.HorizontalFlip(p=1.0), norm],
            bbox_params=bbox_params,
        ),
        # Slight brightness increase (sun simulation)
        A.Compose(
            [base, A.RandomBrightnessContrast(
                brightness_limit=(0.15, 0.15),
                contrast_limit=0, p=1.0,
            ), norm],
            bbox_params=bbox_params,
        ),
        # Slight brightness decrease (shadow simulation)
        A.Compose(
            [base, A.RandomBrightnessContrast(
                brightness_limit=(-0.15, -0.15),
                contrast_limit=0, p=1.0,
            ), norm],
            bbox_params=bbox_params,
        ),
    ]
