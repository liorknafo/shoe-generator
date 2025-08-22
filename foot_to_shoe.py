import argparse
import math
from typing import Tuple

import cv2
import numpy as np
from stl import mesh


def find_foot_keypoints(image_path: str) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Detect heel and big toe from an image.

    Returns (heel, toe, length_px, width_px, angle_radians).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure foot is white on black background
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No contours found in image")
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = eigvecs[:, order[1]]
    projections = centered @ major
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    heel = pts[min_idx]
    toe = pts[max_idx]
    length_px = float(np.linalg.norm(toe - heel))
    widths = centered @ minor
    width_px = float(widths.max() - widths.min())
    angle = math.atan2(major[1], major[0])
    return heel, toe, length_px, width_px, angle


def scale_stl(
    stl_path: str,
    output_path: str,
    target_length_mm: float,
    target_width_mm: float,
    foot_angle: float,
) -> None:
    """Scale and rotate an STL mesh to match the target size."""
    shoe = mesh.Mesh.from_file(stl_path)
    vertices = shoe.vectors.reshape(-1, 3)
    centroid = vertices.mean(axis=0)
    vertices -= centroid
    xy = vertices[:, :2]
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    angle_mesh = math.atan2(major[1], major[0])
    rot_to_x = -angle_mesh
    rot_mat = np.array(
        [
            [math.cos(rot_to_x), -math.sin(rot_to_x), 0],
            [math.sin(rot_to_x), math.cos(rot_to_x), 0],
            [0, 0, 1],
        ]
    )
    vertices[:] = vertices @ rot_mat.T
    cur_length = vertices[:, 0].max() - vertices[:, 0].min()
    cur_width = vertices[:, 1].max() - vertices[:, 1].min()
    len_scale = target_length_mm / cur_length
    wid_scale = target_width_mm / cur_width
    scale_mat = np.diag([len_scale, wid_scale, (len_scale + wid_scale) / 2])
    vertices[:] = vertices @ scale_mat
    rot_back = foot_angle
    rot_mat2 = np.array(
        [
            [math.cos(rot_back), -math.sin(rot_back), 0],
            [math.sin(rot_back), math.cos(rot_back), 0],
            [0, 0, 1],
        ]
    )
    vertices[:] = vertices @ rot_mat2.T
    vertices[:] += centroid
    shoe.vectors = vertices.reshape(-1, 3, 3)
    shoe.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale a shoe STL using a foot photo")
    parser.add_argument("--image", required=True, help="Path to foot image")
    parser.add_argument(
        "--foot-length-mm", type=float, required=True, help="Actual foot length in millimetres"
    )
    parser.add_argument("--stl", required=True, help="Path to shoe STL file")
    parser.add_argument("--output", required=True, help="Output path for scaled STL")
    args = parser.parse_args()
    heel, toe, length_px, width_px, angle = find_foot_keypoints(args.image)
    mm_per_px = args.foot_length_mm / length_px
    width_mm = width_px * mm_per_px
    scale_stl(args.stl, args.output, args.foot_length_mm, width_mm, angle)
    print(f"Heel: {heel}")
    print(f"Big toe: {toe}")
    print(f"Pixel length: {length_px:.2f}")
    print(f"mm per pixel: {mm_per_px:.4f}")
    print(f"Estimated foot width (mm): {width_mm:.2f}")
    print(f"Scaled STL saved to {args.output}")


if __name__ == "__main__":
    main()
