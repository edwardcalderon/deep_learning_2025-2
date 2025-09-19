"""
Circular mask exercise using PIL and NumPy.

Steps implemented:
1) Read the image and convert it to double precision (float64).
2) Create a zero matrix (mask) with the same dimensions.
3) Modify the mask to contain 1's in a circle of radius 150 centered in the image.
4) Multiply the image by the mask (black outside the circle).
5) Modify to keep outside pixels visible at half intensity.

When run as a script, saves intermediate and final images and also displays them.
When imported, exposes a function `process_image` that returns the arrays for programmatic use.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def read_image_as_float64_gray(image_path: str) -> np.ndarray:
    """Read an image with PIL, convert to grayscale, and return as float64 array (0-255 scale)."""
    img = Image.open(image_path).convert("L")  # Grayscale
    arr = np.asarray(img, dtype=np.float64)    # float64, values in [0, 255]
    return arr


def create_circular_mask(h: int, w: int, radius: int) -> np.ndarray:
    """Create a circular mask with ones inside the circle of given radius, centered in the image.

    Args:
        h: image height (rows)
        w: image width (cols)
        radius: radius of the circle in pixels
    Returns:
        mask: float64 array of shape (h, w) with 1.0 inside circle and 0.0 outside
    """
    cy, cx = h / 2.0, w / 2.0  # center (row, col)
    y_indices, x_indices = np.indices((h, w))
    dist2 = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
    mask = (dist2 < (radius ** 2)).astype(np.float64)
    return mask


def apply_mask_black_outside(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply image by binary mask; outside becomes black."""
    return img * mask


def apply_mask_half_intensity_outside(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep pixels outside the circle visible at half intensity.

    Equivalent to a weighted mask where inside=1.0 and outside=0.5:
    result = img * (mask * 1.0 + (1 - mask) * 0.5)
    """
    weighted = mask * 1.0 + (1.0 - mask) * 0.5
    return img * weighted


def process_image(image_path: str, radius: int = 150) -> Dict[str, np.ndarray]:
    """Run the full pipeline and return intermediate arrays.

    Returns a dict with keys:
      - image_float64
      - mask_binary
      - result_black_outside
      - result_half_outside
    """
    img = read_image_as_float64_gray(image_path)
    h, w = img.shape
    mask = create_circular_mask(h, w, radius)
    res_black = apply_mask_black_outside(img, mask)
    res_half = apply_mask_half_intensity_outside(img, mask)
    return {
        "image_float64": img,
        "mask_binary": mask,
        "result_black_outside": res_black,
        "result_half_outside": res_half,
    }


def _save_array_as_png(arr: np.ndarray, out_path: str) -> None:
    """Save a float array assumed in [0,255] to PNG (uint8)."""
    arr_clip = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr_clip, mode="L").save(out_path)


def _visualize_steps(data: Dict[str, np.ndarray], title_suffix: str = "") -> None:
    img = data["image_float64"]
    mask = data["mask_binary"]
    res_black = data["result_black_outside"]
    res_half = data["result_half_outside"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Original (float64){title_suffix}")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask (1 inside circle)")
    axes[1].axis("off")

    axes[2].imshow(res_black, cmap="gray")
    axes[2].set_title("Result: Black outside")
    axes[2].axis("off")

    axes[3].imshow(res_half, cmap="gray")
    axes[3].set_title("Result: Half intensity outside")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(here, "lena_gray_512.tif")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    data = process_image(image_path=image_path, radius=150)

    # Save outputs
    _save_array_as_png(data["image_float64"], os.path.join(here, "01_original_float64.png"))
    _save_array_as_png(data["mask_binary"] * 255.0, os.path.join(here, "02_mask_binary.png"))
    _save_array_as_png(data["result_black_outside"], os.path.join(here, "03_result_black_outside.png"))
    _save_array_as_png(data["result_half_outside"], os.path.join(here, "04_result_half_outside.png"))

    # Visualize
    _visualize_steps(data, title_suffix=" (0-255 float)")


if __name__ == "__main__":
    main()
