from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats
from skimage import exposure, filters, measure, morphology

try:
    import pytesseract  # type: ignore
except ImportError:
    pytesseract = None

try:
    import tifffile  # type: ignore
except ImportError:
    tifffile = None

BASE_DIR = Path(__file__).resolve().parent
STREAMLIT_OUTPUT_ROOT = BASE_DIR / "streamlit_outputs"
_env_matlab_source = os.environ.get("MATLAB_SOURCE_DIR", "").strip()
if _env_matlab_source:
    DEFAULT_MATLAB_SOURCE_DIR = Path(_env_matlab_source).expanduser()
else:
    DEFAULT_MATLAB_SOURCE_DIR = (BASE_DIR / "matlab_source").resolve()


def _resolve_matlab_executable() -> str | None:
    # 1) Explicit override wins.
    env_exe = os.environ.get("MATLAB_EXE", "").strip()
    if env_exe and Path(env_exe).exists():
        return env_exe

    # 2) PATH lookup.
    exe = shutil.which("matlab")
    if exe:
        return exe

    # 3) Common macOS app install locations.
    candidates: list[Path] = []
    for root in (Path("/Applications"), Path.home() / "Applications"):
        if root.exists():
            for app in sorted(root.glob("MATLAB*.app")):
                candidates.append(app / "bin" / "matlab")

    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


@dataclass
class ProcessedImageResult:
    filename: str
    original_image: np.ndarray
    micro_image: np.ndarray
    meta_strip: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray
    calibration: dict[str, Any]
    statistics: dict[str, Any]
    particles: list[dict[str, Any]]


def _set_tesseract_cmd_from_env() -> None:
    if pytesseract is None:
        return
    tesseract_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img[:, :, 0]
    return img


def _remove_small_objects_compat(ar: np.ndarray, min_size: int, connectivity: int = 1) -> np.ndarray:
    threshold = max(0, int(min_size) - 1)
    return morphology.remove_small_objects(ar, max_size=threshold, connectivity=connectivity)


def split_sem_image(img: np.ndarray, strip_ratio: float = 0.073) -> tuple[np.ndarray, np.ndarray]:
    gray = _to_gray(img)
    h = gray.shape[0]
    split_idx = int((1.0 - strip_ratio) * h)
    split_idx = max(1, min(split_idx, h - 1))
    micro = gray[:split_idx, :]
    meta = gray[split_idx:, :]
    return micro, meta


def detect_scale_bar(meta_gray: np.ndarray, debug: bool = False) -> tuple[int, int, list[Any]]:
    h, w = meta_gray.shape
    th = filters.threshold_otsu(meta_gray)

    best_score = -1.0
    best_segments = None

    for polarity_name, bw in [("bright", meta_gray > th), ("dark", meta_gray < th)]:
        bw_clean = _remove_small_objects_compat(bw, min_size=30)
        labels = measure.label(bw_clean)
        regions = measure.regionprops(labels)

        segments = []
        for r in regions:
            minr, minc, maxr, maxc = r.bbox
            height = maxr - minr
            width = maxc - minc
            cy = 0.5 * (minr + maxr)
            aspect_ratio = width / height if height > 0 else 0.0

            if (
                width > 0.02 * w
                and height < 0.35 * h
                and aspect_ratio > 2.0
                and minc > 0.40 * w
                and 0.15 * h < cy < 0.85 * h
            ):
                segments.append(r)
                if debug:
                    print(
                        f"Candidate segment: width={width}, height={height}, "
                        f"aspect={aspect_ratio:.1f}, x=[{minc},{maxc}], y=[{minr},{maxr}]"
                    )

        if not segments:
            continue

        x_start = min(r.bbox[1] for r in segments)
        x_end = max(r.bbox[3] for r in segments)
        span = x_end - x_start
        y_positions = [(r.bbox[0] + r.bbox[2]) / 2 for r in segments]
        y_std = np.std(y_positions)
        alignment_bonus = 2.0 if y_std < 0.1 * h else 1.0
        score = span * len(segments) * alignment_bonus

        if debug:
            print(
                f"{polarity_name} polarity: {len(segments)} segments, "
                f"span={span}px, y_std={y_std:.1f}, score={score:.1f}"
            )

        if score > best_score:
            best_score = score
            best_segments = segments

    if best_segments is None:
        raise RuntimeError("Scale bar not detected - try adjusting strip ratio")

    x_start = min(r.bbox[1] for r in best_segments)
    x_end = max(r.bbox[3] for r in best_segments)
    return x_start, x_end, best_segments


def extract_scale_value_ocr(text_roi: np.ndarray, fallback_um: float = 100.0, debug: bool = False) -> tuple[float, str]:
    _set_tesseract_cmd_from_env()
    if pytesseract is None:
        return float(fallback_um), (
            f"OCR unavailable (pytesseract missing) -> Fallback: {fallback_um} um "
            "(MANUAL CHECK REQUIRED)"
        )

    roi = text_roi.astype(np.float32)
    roi = (roi - roi.min()) / (roi.max() - roi.min() + 1e-6)
    roi = (roi * 255).astype(np.uint8)
    roi_inv = 255 - roi

    ocr_results: list[tuple[str, str]] = []
    config = "--psm 7 -c tessedit_char_whitelist=0123456789.um"

    ocr_results.append(("inverted-psm7", pytesseract.image_to_string(roi_inv, config=config)))
    ocr_results.append(("inverted-psm8", pytesseract.image_to_string(roi_inv, config=config.replace("--psm 7", "--psm 8"))))
    ocr_results.append(("original-psm7", pytesseract.image_to_string(roi, config=config)))

    candidates: list[float] = []
    for _, text in ocr_results:
        for num_str in re.findall(r"\d+\.?\d*", text):
            try:
                value = float(num_str)
            except ValueError:
                continue
            if 1 <= value <= 1000:
                candidates.append(value)

    if candidates:
        scale_value_um = float(Counter(candidates).most_common(1)[0][0])
        ocr_summary = f"Found: {scale_value_um} um (from {len(candidates)} candidates)"
    else:
        scale_value_um = float(fallback_um)
        ocr_summary = f"OCR failed -> Fallback: {fallback_um} um (MANUAL CHECK REQUIRED)"

    if debug:
        print(ocr_results)

    return scale_value_um, ocr_summary


def calibrate_single_image_from_strip(
    meta_img: np.ndarray,
    fallback_um: float = 100.0,
    debug: bool = False,
    manual_scale_um: float | None = None,
) -> dict[str, Any]:
    meta_gray = _to_gray(meta_img)
    h, _ = meta_gray.shape

    x_start, x_end, bar_segments = detect_scale_bar(meta_gray, debug=debug)
    bar_pixel_length = x_end - x_start
    if bar_pixel_length <= 0:
        raise RuntimeError("Invalid bar length detected")

    bar_minr = min(r.bbox[0] for r in bar_segments)
    bar_maxr = max(r.bbox[2] for r in bar_segments)
    bar_height = bar_maxr - bar_minr

    vertical_margin = max(int(1.5 * bar_height), 20)
    y0 = max(bar_minr - vertical_margin, 0)
    y1 = min(bar_maxr + vertical_margin, h)

    text_roi = meta_gray[y0:y1, x_start:x_end]

    if manual_scale_um is not None and manual_scale_um > 0:
        scale_value_um = float(manual_scale_um)
        ocr_summary = f"Manual override: {scale_value_um} um"
    else:
        scale_value_um, ocr_summary = extract_scale_value_ocr(text_roi, fallback_um=fallback_um, debug=debug)

    microns_per_pixel = scale_value_um / bar_pixel_length

    return {
        "microns_per_pixel": float(microns_per_pixel),
        "scale_value_um": float(scale_value_um),
        "bar_pixel_length": int(bar_pixel_length),
        "ocr_summary": ocr_summary,
    }


def segment_graphite(
    img: np.ndarray,
    mean_intensity: float | None = None,
    min_diameter_um: float = 5.0,
    min_circularity: float = 0.2,
    microns_per_pixel: float = 1.0,
) -> tuple[np.ndarray, str]:
    gray = _to_gray(img)

    if mean_intensity is None:
        mean_intensity = float(np.mean(gray))

    min_radius_px = (min_diameter_um / 2.0) / max(microns_per_pixel, 1e-9)
    min_area_px = int(np.pi * min_radius_px**2)

    if mean_intensity < 90:
        strategy = "advanced"

        img_clahe = exposure.equalize_adapthist(gray, clip_limit=0.05)
        img_clahe = (img_clahe * 255).astype(np.uint8)

        img_smooth = filters.gaussian(img_clahe, sigma=1.5)
        img_smooth = (img_smooth * 255 / max(img_smooth.max(), 1e-6)).astype(np.uint8)

        local_thresh = filters.threshold_local(img_smooth, block_size=151, offset=15, method="gaussian")
        binary = img_smooth < local_thresh

        binary = _remove_small_objects_compat(binary, min_size=100)
        binary = ndimage.binary_fill_holes(binary)
        binary = morphology.opening(binary, morphology.disk(4))

        labels = measure.label(binary)
        regions = measure.regionprops(labels)
        final_mask = np.zeros_like(binary, dtype=bool)

        for r in regions:
            if r.perimeter > 0:
                circularity = 4 * np.pi * r.area / (r.perimeter**2)
                if circularity > 0.6 and 150 < r.area < 8000:
                    final_mask[labels == r.label] = True
    else:
        strategy = "simple"
        otsu_thresh = filters.threshold_otsu(gray)
        binary = gray < otsu_thresh
        binary = _remove_small_objects_compat(binary, min_size=max(20, min_area_px))
        binary = ndimage.binary_fill_holes(binary)

        labels = measure.label(binary)
        regions = measure.regionprops(labels)
        final_mask = np.zeros_like(binary, dtype=bool)

        for r in regions:
            if r.perimeter > 0:
                circularity = 4 * np.pi * r.area / (r.perimeter**2)
                if circularity >= min_circularity and r.area >= min_area_px:
                    final_mask[labels == r.label] = True

    return final_mask, strategy


def extract_graphite_statistics(mask: np.ndarray, microns_per_pixel: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    px_to_um = float(microns_per_pixel)
    px2_to_um2 = px_to_um**2

    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    a_graphite_px = int(np.sum(mask))
    a_total_px = int(mask.size)
    phase_fraction = a_graphite_px / a_total_px if a_total_px else 0.0

    particle_data: list[dict[str, Any]] = []
    for r in regions:
        area_um2 = r.area * px2_to_um2
        d_eq_um = 2 * np.sqrt(area_um2 / np.pi)
        aspect_ratio = r.axis_major_length / r.axis_minor_length if r.axis_minor_length > 0 else np.nan
        circularity = 4 * np.pi * r.area / (r.perimeter**2) if r.perimeter > 0 else np.nan

        particle_data.append(
            {
                "area_um2": float(area_um2),
                "diameter_eq_um": float(d_eq_um),
                "aspect_ratio": float(aspect_ratio) if not np.isnan(aspect_ratio) else np.nan,
                "circularity": float(circularity) if not np.isnan(circularity) else np.nan,
                "centroid_y_px": float(r.centroid[0]),
                "centroid_x_px": float(r.centroid[1]),
            }
        )

    if particle_data:
        diameters = [p["diameter_eq_um"] for p in particle_data]
        aspects = [p["aspect_ratio"] for p in particle_data if not np.isnan(p["aspect_ratio"])]
        circs = [p["circularity"] for p in particle_data if not np.isnan(p["circularity"])]

        a_image_um2 = a_total_px * px2_to_um2
        number_density = len(regions) / a_image_um2 if a_image_um2 > 0 else 0.0

        stats_dict: dict[str, Any] = {
            "phase_fraction": float(phase_fraction),
            "n_particles": int(len(regions)),
            "number_density_um2": float(number_density),
            "diameter_um_mean": float(np.mean(diameters)),
            "diameter_um_std": float(np.std(diameters)),
            "diameter_um_min": float(np.min(diameters)),
            "diameter_um_max": float(np.max(diameters)),
            "aspect_ratio_mean": float(np.mean(aspects)) if aspects else np.nan,
            "aspect_ratio_std": float(np.std(aspects)) if aspects else np.nan,
            "circularity_mean": float(np.mean(circs)) if circs else np.nan,
            "circularity_std": float(np.std(circs)) if circs else np.nan,
        }
    else:
        stats_dict = {
            "phase_fraction": 0.0,
            "n_particles": 0,
            "number_density_um2": 0.0,
            "diameter_um_mean": np.nan,
            "diameter_um_std": np.nan,
            "diameter_um_min": np.nan,
            "diameter_um_max": np.nan,
            "aspect_ratio_mean": np.nan,
            "aspect_ratio_std": np.nan,
            "circularity_mean": np.nan,
            "circularity_std": np.nan,
        }

    return stats_dict, particle_data


def fit_lognormal(diameters: np.ndarray) -> dict[str, Any]:
    diameters = np.asarray(diameters, dtype=float)
    diameters = diameters[np.isfinite(diameters)]
    diameters = diameters[diameters > 0]

    if diameters.size < 3:
        return {
            "success": False,
            "message": "Not enough positive diameters for lognormal fit (need at least 3).",
        }

    shape, loc, scale = stats.lognorm.fit(diameters, floc=0)
    x = np.linspace(float(diameters.min()), float(diameters.max()), 500)
    pdf = stats.lognorm.pdf(x, shape, loc, scale)
    ks_stat, p_value = stats.kstest(diameters, "lognorm", args=(shape, loc, scale))

    return {
        "success": True,
        "shape": float(shape),
        "loc": float(loc),
        "scale": float(scale),
        "mu": float(np.log(scale)),
        "sigma": float(shape),
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "x": x,
        "pdf": pdf,
    }


def _read_uploaded_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    ext = Path(filename).suffix.lower()
    if ext in {".tif", ".tiff"} and tifffile is not None:
        return tifffile.imread(io.BytesIO(file_bytes))
    with Image.open(io.BytesIO(file_bytes)) as pil_img:
        return np.array(pil_img)


def _write_mask_tiff(mask: np.ndarray, mask_path: Path) -> None:
    mask_u8 = (mask.astype(np.uint8) * 255)
    if tifffile is not None:
        tifffile.imwrite(mask_path, mask_u8)
    else:
        Image.fromarray(mask_u8).save(mask_path, format="TIFF")


def _build_overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray_u8 = gray.astype(np.uint8)
    rgb = np.dstack([gray_u8, gray_u8, gray_u8]).astype(np.float32)
    rgb[..., 0] = np.where(mask, np.clip(rgb[..., 0] + 120, 0, 255), rgb[..., 0])
    rgb[..., 1] = np.where(mask, rgb[..., 1] * 0.5, rgb[..., 1])
    rgb[..., 2] = np.where(mask, rgb[..., 2] * 0.5, rgb[..., 2])
    return rgb.astype(np.uint8)


def _crop_bottom_pixels(img: np.ndarray, bottom_px: int) -> tuple[np.ndarray, np.ndarray]:
    gray = _to_gray(img)
    if bottom_px <= 0:
        return gray, np.zeros((1, gray.shape[1]), dtype=gray.dtype)
    h = gray.shape[0]
    crop = max(0, min(int(bottom_px), h - 1))
    if crop == 0:
        return gray, np.zeros((1, gray.shape[1]), dtype=gray.dtype)
    return gray[:-crop, :], gray[-crop:, :]


def process_uploaded_image(
    file_bytes: bytes,
    filename: str,
    strip_ratio: float,
    fallback_um: float,
    manual_scale_um: float | None = None,
    manual_microns_per_pixel: float | None = None,
    exclude_bottom_px: int = 0,
    min_diameter_um: float = 5.0,
    min_circularity: float = 0.2,
) -> dict[str, Any]:
    img = _read_uploaded_bytes(file_bytes, filename)
    if manual_microns_per_pixel is not None and manual_microns_per_pixel > 0:
        micro_img, meta_strip = _crop_bottom_pixels(img, exclude_bottom_px)
        calibration = {
            "microns_per_pixel": float(manual_microns_per_pixel),
            "scale_value_um": np.nan,
            "bar_pixel_length": np.nan,
            "ocr_summary": f"Manual override: {float(manual_microns_per_pixel):.6f} um/px",
        }
    else:
        micro_img, meta_strip = split_sem_image(img, strip_ratio=strip_ratio)
        if exclude_bottom_px > 0:
            micro_img, removed_strip = _crop_bottom_pixels(micro_img, exclude_bottom_px)
            meta_strip = np.vstack([_to_gray(removed_strip), _to_gray(meta_strip)])

        calibration = calibrate_single_image_from_strip(
            meta_strip,
            fallback_um=fallback_um,
            manual_scale_um=manual_scale_um,
            debug=False,
        )

    mean_intensity = float(np.mean(micro_img))
    mask, strategy = segment_graphite(
        micro_img,
        mean_intensity=mean_intensity,
        min_diameter_um=min_diameter_um,
        min_circularity=min_circularity,
        microns_per_pixel=calibration["microns_per_pixel"],
    )
    stats_dict, particles = extract_graphite_statistics(mask, calibration["microns_per_pixel"])

    image_name = Path(filename).stem
    stats_dict.update(
        {
            "image_name": image_name,
            "mean_intensity": mean_intensity,
            "strategy": strategy,
            "microns_per_pixel": calibration["microns_per_pixel"],
        }
    )

    for p in particles:
        p["image_name"] = image_name

    overlay = _build_overlay(_to_gray(micro_img), mask)

    result = ProcessedImageResult(
        filename=filename,
        original_image=_to_gray(img),
        micro_image=_to_gray(micro_img),
        meta_strip=_to_gray(meta_strip),
        mask=mask,
        overlay=overlay,
        calibration={
            "image_name": image_name,
            "microns_per_pixel": calibration["microns_per_pixel"],
            "scale_value_um": calibration["scale_value_um"],
            "bar_pixel_length": calibration["bar_pixel_length"],
            "ocr_summary": calibration["ocr_summary"],
        },
        statistics=stats_dict,
        particles=particles,
    )
    return result.__dict__


def _make_timestamped_output_dir() -> Path:
    ts = datetime.now().strftime("%d-%b-%Y %H_%M_%S")
    out_dir = STREAMLIT_OUTPUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _choose_hist_bins(diameters: np.ndarray) -> int:
    values = np.asarray(diameters, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    n = values.size
    if n <= 1:
        return 1
    if n < 12:
        return max(4, n // 2)

    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    data_range = float(values.max() - values.min())

    if iqr > 0 and data_range > 0:
        bin_width = 2.0 * iqr / np.cbrt(n)
        if bin_width > 0:
            bins = int(np.ceil(data_range / bin_width))
            return max(8, min(20, bins))

    bins = int(np.ceil(np.sqrt(n)))
    return max(8, min(20, bins))


def _save_distribution_plot(diameters: np.ndarray, fit: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    fig_paths: dict[str, Path] = {}
    bins = _choose_hist_bins(diameters)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(diameters, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Equivalent Diameter (um)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Pooled Size Distribution (Linear, {bins} bins)")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(diameters, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Equivalent Diameter (um)")
    axes[1].set_ylabel("Count (log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Pooled Size Distribution (Log, {bins} bins)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    pooled_path = out_dir / "pooled_size_distribution.png"
    fig.savefig(pooled_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig_paths["pooled_size_distribution"] = pooled_path

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(diameters, bins=bins, density=True, alpha=0.6, edgecolor="black")
    if fit.get("success"):
        ax2.plot(fit["x"], fit["pdf"], "r", linewidth=2, label="Lognormal Fit")
        ax2.legend()
    ax2.set_xlabel("Equivalent Diameter (um)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title(f"Lognormal Fit to Graphite Size Distribution ({bins} bins)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fit_path = out_dir / "lognormal_fit.png"
    fig2.savefig(fit_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    fig_paths["lognormal_fit"] = fit_path

    return fig_paths


def _compute_pooled_graphite_inputs(statistics_df: pd.DataFrame, particle_df: pd.DataFrame) -> dict[str, float]:
    if statistics_df.empty or particle_df.empty:
        raise RuntimeError("Need non-empty statistics and particle data for 3D generation.")

    graphite_vf = float(statistics_df["phase_fraction"].mean())
    aspect_vals = particle_df["aspect_ratio"].dropna().to_numpy(dtype=float)
    graphite_aspect_ratio = float(np.mean(aspect_vals)) if aspect_vals.size else 1.0

    diameters = particle_df["diameter_eq_um"].dropna().to_numpy(dtype=float)
    diameters = diameters[diameters > 0]
    if diameters.size < 3:
        raise RuntimeError("Need at least 3 positive diameters for lognormal fit.")

    shape, loc, scale = stats.lognorm.fit(diameters, floc=0)
    mu_log = float(np.log(scale))
    sigma_log = float(shape)

    return {
        "graphite_vf": graphite_vf,
        "graphite_aspect_ratio": graphite_aspect_ratio,
        "mu_log": mu_log,
        "sigma_log": sigma_log,
    }


def _estimate_mesh_length_um(
    graphite_vf: float,
    graphite_aspect_ratio: float,
    mu_log: float,
    sigma_log: float,
    target_graphite_particles: int,
) -> float:
    e_d3 = float(np.exp(3 * mu_log + 4.5 * (sigma_log**2)))
    e_r3 = e_d3 / 8.0
    v_mean_particle = (4.0 * np.pi / 3.0) * graphite_aspect_ratio * e_r3
    mesh_length = ((target_graphite_particles * v_mean_particle) / max(graphite_vf, 1e-9)) ** (1.0 / 3.0)
    return float(mesh_length)


def _generate_graphite_volume(
    graphite_vf: float,
    graphite_aspect_ratio: float,
    mu_log: float,
    sigma_log: float,
    target_graphite_particles: int,
    voxel_size_um: float,
    max_particles: int,
    seed: int | None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)

    mesh_length_um = _estimate_mesh_length_um(
        graphite_vf,
        graphite_aspect_ratio,
        mu_log,
        sigma_log,
        target_graphite_particles,
    )

    n_vox = int(np.clip(np.round(mesh_length_um / max(voxel_size_um, 1e-6)), 40, 120))
    volume = np.zeros((n_vox, n_vox, n_vox), dtype=bool)

    particles: list[dict[str, float]] = []
    center_arr: list[tuple[float, float, float, float]] = []

    target_particles = int(min(target_graphite_particles, max_particles))
    max_attempts_total = target_particles * 220
    attempts = 0

    while len(particles) < target_particles and attempts < max_attempts_total:
        attempts += 1

        d_um = float(stats.lognorm.rvs(s=sigma_log, scale=np.exp(mu_log), random_state=rng))
        d_um = float(np.clip(d_um, 0.3, np.exp(mu_log + 3.0 * sigma_log)))

        r_um = d_um / 2.0
        rx_um = r_um * graphite_aspect_ratio
        ry_um = r_um
        rz_um = r_um

        rx = rx_um / voxel_size_um
        ry = ry_um / voxel_size_um
        rz = rz_um / voxel_size_um
        rs = max(rx, ry, rz)

        if rs < 1.0 or rs > n_vox / 6.0:
            continue

        cx = float(rng.uniform(rs + 1, n_vox - rs - 1))
        cy = float(rng.uniform(rs + 1, n_vox - rs - 1))
        cz = float(rng.uniform(rs + 1, n_vox - rs - 1))

        overlaps = False
        for ex, ey, ez, ers in center_arr:
            if (cx - ex) ** 2 + (cy - ey) ** 2 + (cz - ez) ** 2 < (rs + ers) ** 2:
                overlaps = True
                break
        if overlaps:
            continue

        xmin = max(0, int(np.floor(cx - rx)))
        xmax = min(n_vox - 1, int(np.ceil(cx + rx)))
        ymin = max(0, int(np.floor(cy - ry)))
        ymax = min(n_vox - 1, int(np.ceil(cy + ry)))
        zmin = max(0, int(np.floor(cz - rz)))
        zmax = min(n_vox - 1, int(np.ceil(cz + rz)))

        zz, yy, xx = np.ogrid[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1]
        ell = ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 + ((zz - cz) / max(rz, 1e-6)) ** 2 <= 1.0

        subvol = volume[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1]
        if np.any(subvol & ell):
            continue

        volume[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1] |= ell
        center_arr.append((cx, cy, cz, rs))
        particles.append(
            {
                "cx_vox": cx,
                "cy_vox": cy,
                "cz_vox": cz,
                "rx_vox": rx,
                "ry_vox": ry,
                "rz_vox": rz,
                "diameter_um": d_um,
            }
        )

        if np.mean(volume) >= graphite_vf * 0.98:
            break

    packed_graphite_vf = float(np.mean(volume))
    packed_matrix_vf = float(1.0 - packed_graphite_vf)

    summary = {
        "mesh_length_um": float(n_vox * voxel_size_um),
        "mesh_n_voxels": int(n_vox),
        "target_graphite_particles": int(target_graphite_particles),
        "packed_graphite_particles": int(len(particles)),
        "target_graphite_vf": float(graphite_vf),
        "packed_graphite_vf": packed_graphite_vf,
        "packed_matrix_vf": packed_matrix_vf,
        "mu_log": float(mu_log),
        "sigma_log": float(sigma_log),
        "graphite_aspect_ratio": float(graphite_aspect_ratio),
        "voxel_size_um": float(voxel_size_um),
    }

    return volume, pd.DataFrame(particles), summary


def _save_3d_artifacts(out_dir: Path, volume: np.ndarray, particles_df: pd.DataFrame, summary: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}

    npy_path = out_dir / "graphite_volume.npy"
    np.save(npy_path, volume.astype(np.uint8))
    paths["graphite_volume_npy"] = str(npy_path)

    particles_csv = out_dir / "graphite_particles_3d.csv"
    particles_df.to_csv(particles_csv, index=False)
    paths["graphite_particles_3d_csv"] = str(particles_csv)

    summary_csv = out_dir / "graphite_only_results_matrix.csv"
    row = {
        "iter": 1,
        "mesh_length_um": summary["mesh_length_um"],
        "expected_graphite_n": summary["target_graphite_particles"],
        "packed_graphite_n": summary["packed_graphite_particles"],
        "target_graphite_vf": summary["target_graphite_vf"],
        "packed_graphite_vf": summary["packed_graphite_vf"],
        "packed_matrix_vf": summary["packed_matrix_vf"],
    }
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    paths["graphite_only_results_matrix_csv"] = str(summary_csv)

    inputs_txt = out_dir / "graphite_only_inputs.txt"
    with open(inputs_txt, "w", encoding="utf-8") as f:
        f.write("Graphite-only run inputs\n")
        f.write(f"vf_graphite = {summary['target_graphite_vf']:.6f}\n")
        f.write(f"aspect_ratio_mean = {summary['graphite_aspect_ratio']:.6f}\n")
        f.write("dist_type = LogNormal\n")
        f.write(f"mu_log = {summary['mu_log']:.6f}\n")
        f.write(f"sigma_log = {summary['sigma_log']:.6f}\n")
        f.write(f"mesh_length_um = {summary['mesh_length_um']:.6f}\n")
        f.write(f"voxel_size_um = {summary['voxel_size_um']:.6f}\n")
    paths["graphite_only_inputs_txt"] = str(inputs_txt)

    zmid, ymid, xmid = np.array(volume.shape) // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[zmid, :, :], cmap="gray")
    axes[0].set_title("XY mid-slice")
    axes[1].imshow(volume[:, ymid, :], cmap="gray")
    axes[1].set_title("XZ mid-slice")
    axes[2].imshow(volume[:, :, xmid], cmap="gray")
    axes[2].set_title("YZ mid-slice")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    slice_png = out_dir / "graphite_volume_mid_slices.png"
    fig.savefig(slice_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["graphite_volume_mid_slices_png"] = str(slice_png)

    return paths


def _matlab_escape(path_text: str) -> str:
    return path_text.replace("'", "''")


def _build_matlab_graphite_runner_script(
    script_path: Path,
    source_dir: Path,
    matlab_out_dir: Path,
    graphite_vf: float,
    graphite_aspect_ratio: float,
    mu_log: float,
    sigma_log: float,
    target_graphite_particles: int,
    n_iter: int,
) -> None:
    source_esc = _matlab_escape(str(source_dir))
    out_esc = _matlab_escape(str(matlab_out_dir))
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("clc; clear; close all;\n")
        f.write("set(0, 'DefaultFigureVisible', 'off');\n")
        f.write(f"source_dir = '{source_esc}';\n")
        f.write(f"path_to_results = '{out_esc}';\n")
        f.write("if exist(path_to_results,'dir') ~= 7, mkdir(path_to_results); end\n")
        f.write("if path_to_results(end) ~= filesep, path_to_results = [path_to_results filesep]; end\n")
        f.write("addpath(source_dir);\n")
        f.write("cd(source_dir);\n")
        f.write("graphite_only_cfg = struct();\n")
        f.write(f"graphite_only_cfg.graphite_vf = {graphite_vf:.12f};\n")
        f.write(f"graphite_only_cfg.graphite_aspect_ratio = {graphite_aspect_ratio:.12f};\n")
        f.write(f"graphite_only_cfg.mu_log = {mu_log:.12f};\n")
        f.write(f"graphite_only_cfg.sigma_log = {sigma_log:.12f};\n")
        f.write(f"graphite_only_cfg.target_graphite_particles = {int(target_graphite_particles)};\n")
        f.write(f"graphite_only_cfg.n_iter = {int(max(1, n_iter))};\n")
        f.write("graphite_only_cfg.path_to_results = path_to_results;\n")
        f.write("graphite_only_cfg.random_seed = [];\n")
        f.write("graphite_only_main;\n")
        f.write("close all;\n")


def _voxelize_matlab_ellipsoids(
    ellipsoids_df: pd.DataFrame,
    mesh_length_um: float,
    voxel_size_um: float,
) -> np.ndarray:
    n_vox = int(np.clip(np.round(mesh_length_um / max(voxel_size_um, 1e-6)), 40, 140))
    volume = np.zeros((n_vox, n_vox, n_vox), dtype=bool)
    if ellipsoids_df.empty:
        return volume

    has_orientation = {"phi1_rad", "Phi_rad", "phi2_rad"}.issubset(ellipsoids_df.columns)

    for _, row in ellipsoids_df.iterrows():
        rx = float(row["rx_um"]) / voxel_size_um
        ry = float(row["ry_um"]) / voxel_size_um
        rz = float(row["rz_um"]) / voxel_size_um
        cx = float(row["cx_um"]) / voxel_size_um
        cy = float(row["cy_um"]) / voxel_size_um
        cz = float(row["cz_um"]) / voxel_size_um

        rx = max(rx, 0.5)
        ry = max(ry, 0.5)
        rz = max(rz, 0.5)

        # Use a spherical bound so rotated ellipsoids are fully covered by the candidate box.
        rs = max(rx, ry, rz)
        xmin = max(0, int(np.floor(cx - rs)))
        xmax = min(n_vox - 1, int(np.ceil(cx + rs)))
        ymin = max(0, int(np.floor(cy - rs)))
        ymax = min(n_vox - 1, int(np.ceil(cy + rs)))
        zmin = max(0, int(np.floor(cz - rs)))
        zmax = min(n_vox - 1, int(np.ceil(cz + rs)))

        if xmin > xmax or ymin > ymax or zmin > zmax:
            continue

        zz, yy, xx = np.ogrid[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1]

        dx = xx - cx
        dy = yy - cy
        dz = zz - cz

        if has_orientation:
            phi1 = float(row["phi1_rad"])
            Phi = float(row["Phi_rad"])
            phi2 = float(row["phi2_rad"])

            c1, s1 = np.cos(phi1), np.sin(phi1)
            c, s = np.cos(Phi), np.sin(Phi)
            c2, s2 = np.cos(phi2), np.sin(phi2)

            rz1 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
            rxm = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            rz2 = np.array([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
            rot = rz1 @ rxm @ rz2
            # Transform world offset into ellipsoid local coordinates (inverse rotation).
            inv = rot.T
            lx = inv[0, 0] * dx + inv[0, 1] * dy + inv[0, 2] * dz
            ly = inv[1, 0] * dx + inv[1, 1] * dy + inv[1, 2] * dz
            lz = inv[2, 0] * dx + inv[2, 1] * dy + inv[2, 2] * dz
            ell = (lx / rx) ** 2 + (ly / ry) ** 2 + (lz / rz) ** 2 <= 1.0
        else:
            ell = (dx / rx) ** 2 + (dy / ry) ** 2 + (dz / rz) ** 2 <= 1.0

        volume[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1] |= ell

    return volume


def _run_matlab_graphite_only(
    out_dir: Path,
    graphite_vf: float,
    graphite_aspect_ratio: float,
    mu_log: float,
    sigma_log: float,
    target_graphite_particles: int,
    voxel_size_um: float,
    matlab_source_dir: Path,
    matlab_iterations: int,
) -> dict[str, Any]:
    matlab_exe = _resolve_matlab_executable()
    if matlab_exe is None:
        return {
            "enabled": True,
            "success": False,
            "message": "MATLAB executable not found. Set MATLAB_EXE or add matlab to PATH.",
        }

    if not matlab_source_dir.exists():
        return {
            "enabled": True,
            "success": False,
            "message": f"MATLAB source directory not found: {matlab_source_dir}",
        }

    matlab_out_dir = out_dir / "matlab_graphite_only"
    matlab_out_dir.mkdir(parents=True, exist_ok=True)

    runner_script = matlab_out_dir / "run_graphite_only.m"
    _build_matlab_graphite_runner_script(
        script_path=runner_script,
        source_dir=matlab_source_dir,
        matlab_out_dir=matlab_out_dir,
        graphite_vf=graphite_vf,
        graphite_aspect_ratio=graphite_aspect_ratio,
        mu_log=mu_log,
        sigma_log=sigma_log,
        target_graphite_particles=target_graphite_particles,
        n_iter=matlab_iterations,
    )

    run_expr = f"run('{_matlab_escape(str(runner_script))}')"
    proc = subprocess.run(
        [matlab_exe, "-batch", run_expr],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    stdout_path = matlab_out_dir / "matlab_stdout.txt"
    stderr_path = matlab_out_dir / "matlab_stderr.txt"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        return {
            "enabled": True,
            "success": False,
            "engine": "matlab",
            "message": f"MATLAB run failed with code {proc.returncode}. Check matlab_stderr.txt.",
            "paths": {
                "matlab_stdout": str(stdout_path),
                "matlab_stderr": str(stderr_path),
                "matlab_runner_script": str(runner_script),
            },
        }

    summary_csv = matlab_out_dir / "graphite_only_results_matrix.csv"
    inputs_txt = matlab_out_dir / "graphite_only_inputs.txt"
    ellipsoids_csv = matlab_out_dir / "ellipsoids_data_1.csv"
    packing_png = matlab_out_dir / "packing_1.png"

    if not summary_csv.exists() or not ellipsoids_csv.exists():
        return {
            "enabled": True,
            "success": False,
            "engine": "matlab",
            "message": "MATLAB completed but expected output files were not found.",
            "paths": {
                "matlab_stdout": str(stdout_path),
                "matlab_stderr": str(stderr_path),
                "matlab_runner_script": str(runner_script),
            },
        }

    summary_df = pd.read_csv(summary_csv, header=None)
    if summary_df.empty:
        return {
            "enabled": True,
            "success": False,
            "engine": "matlab",
            "message": "MATLAB summary CSV is empty.",
            "paths": {
                "matlab_stdout": str(stdout_path),
                "matlab_stderr": str(stderr_path),
                "matlab_runner_script": str(runner_script),
                "graphite_only_results_matrix_csv": str(summary_csv),
            },
        }

    if summary_df.shape[1] < 7:
        return {
            "enabled": True,
            "success": False,
            "engine": "matlab",
            "message": "MATLAB summary CSV has unexpected format.",
            "paths": {
                "matlab_stdout": str(stdout_path),
                "matlab_stderr": str(stderr_path),
                "matlab_runner_script": str(runner_script),
                "graphite_only_results_matrix_csv": str(summary_csv),
            },
        }

    # MATLAB writematrix outputs headerless numeric CSV.
    summary_df.columns = [
        "iter",
        "mesh_length_um",
        "expected_graphite_n",
        "packed_graphite_n",
        "target_graphite_vf",
        "packed_graphite_vf",
        "packed_matrix_vf",
        "time_hours",
    ][: summary_df.shape[1]]

    summary_row = summary_df.iloc[0]
    ell = pd.read_csv(ellipsoids_csv, header=None)
    ell.columns = [
        "rx_um",
        "ry_um",
        "rz_um",
        "cx_um",
        "cy_um",
        "cz_um",
        "phi1_rad",
        "Phi_rad",
        "phi2_rad",
    ]

    mesh_length_um = float(summary_row["mesh_length_um"])
    volume = _voxelize_matlab_ellipsoids(ell, mesh_length_um=mesh_length_um, voxel_size_um=voxel_size_um)

    paths = {
        "matlab_stdout": str(stdout_path),
        "matlab_stderr": str(stderr_path),
        "matlab_runner_script": str(runner_script),
        "graphite_only_results_matrix_csv": str(summary_csv),
        "graphite_only_inputs_txt": str(inputs_txt),
        "ellipsoids_data_csv": str(ellipsoids_csv),
    }
    if packing_png.exists():
        paths["packing_png"] = str(packing_png)

    summary = {
        "mesh_length_um": mesh_length_um,
        "target_graphite_particles": int(target_graphite_particles),
        "packed_graphite_particles": int(summary_row["packed_graphite_n"]),
        "target_graphite_vf": float(summary_row["target_graphite_vf"]),
        "packed_graphite_vf": float(summary_row["packed_graphite_vf"]),
        "packed_matrix_vf": float(summary_row["packed_matrix_vf"]),
        "mu_log": float(mu_log),
        "sigma_log": float(sigma_log),
        "graphite_aspect_ratio": float(graphite_aspect_ratio),
        "voxel_size_um": float(voxel_size_um),
    }

    return {
        "enabled": True,
        "success": True,
        "engine": "matlab",
        "summary": summary,
        "paths": paths,
        "volume": volume,
        "particles_df": ell,
    }


def run_batch(
    files: list[Any],
    strip_ratio: float,
    fallback_um: float,
    manual_overrides: dict[str, dict[str, float | int | None]] | None = None,
    min_diameter_um: float = 5.0,
    min_circularity: float = 0.2,
    run_3d: bool = True,
    target_graphite_particles: int = 200,
    voxel_size_um: float = 2.0,
    max_3d_particles: int = 350,
    random_seed: int | None = None,
    three_d_engine: str = "matlab",
    matlab_source_dir: str | None = None,
    matlab_iterations: int = 1,
    fallback_to_python_3d: bool = True,
) -> dict[str, Any]:
    manual_overrides = manual_overrides or {}

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for f in files:
        try:
            if hasattr(f, "read") and hasattr(f, "name"):
                filename = f.name
                file_bytes = f.read()
                if hasattr(f, "seek"):
                    f.seek(0)
            elif isinstance(f, dict) and "name" in f and "bytes" in f:
                filename = str(f["name"])
                file_bytes = f["bytes"]
            else:
                raise ValueError("Unsupported file input type")

            override_cfg = manual_overrides.get(filename, {})
            result = process_uploaded_image(
                file_bytes=file_bytes,
                filename=filename,
                strip_ratio=strip_ratio,
                fallback_um=fallback_um,
                manual_scale_um=(
                    float(override_cfg["manual_scale_um"])
                    if override_cfg.get("manual_scale_um") is not None
                    else None
                ),
                manual_microns_per_pixel=(
                    float(override_cfg["manual_microns_per_pixel"])
                    if override_cfg.get("manual_microns_per_pixel") is not None
                    else None
                ),
                exclude_bottom_px=int(override_cfg.get("exclude_bottom_px", 0) or 0),
                min_diameter_um=min_diameter_um,
                min_circularity=min_circularity,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            errors.append({"filename": getattr(f, "name", "unknown"), "error": str(exc)})

    calibration_df = pd.DataFrame([r["calibration"] for r in results])
    statistics_df = pd.DataFrame([r["statistics"] for r in results])
    particle_df = pd.DataFrame([p for r in results for p in r["particles"]])

    calib_cols = ["image_name", "microns_per_pixel", "scale_value_um", "bar_pixel_length", "ocr_summary"]
    stats_cols = [
        "image_name",
        "microns_per_pixel",
        "mean_intensity",
        "strategy",
        "phase_fraction",
        "n_particles",
        "number_density_um2",
        "diameter_um_mean",
        "diameter_um_std",
        "diameter_um_min",
        "diameter_um_max",
        "aspect_ratio_mean",
        "aspect_ratio_std",
        "circularity_mean",
        "circularity_std",
    ]
    particle_cols = [
        "image_name",
        "diameter_eq_um",
        "area_um2",
        "aspect_ratio",
        "circularity",
        "centroid_x_px",
        "centroid_y_px",
    ]

    if not calibration_df.empty:
        calibration_df = calibration_df.reindex(columns=calib_cols)
    if not statistics_df.empty:
        statistics_df = statistics_df.reindex(columns=stats_cols)
    if not particle_df.empty:
        particle_df = particle_df.reindex(columns=particle_cols)

    out_dir = _make_timestamped_output_dir()

    calibration_csv = out_dir / "calibration_results.csv"
    summary_csv = out_dir / "summary_statistics.csv"
    particle_csv = out_dir / "particle_data.csv"
    calibration_df.to_csv(calibration_csv, index=False)
    statistics_df.to_csv(summary_csv, index=False)
    particle_df.to_csv(particle_csv, index=False)

    masks: dict[str, str] = {}
    for r in results:
        image_name = r["statistics"]["image_name"]
        mask_path = out_dir / f"{image_name}_mask.tif"
        _write_mask_tiff(r["mask"], mask_path)
        masks[image_name] = str(mask_path)

    figures: dict[str, str] = {}
    pooled_fit: dict[str, Any] = {"success": False, "message": "No pooled diameter data available."}
    pooled_summary: dict[str, Any] = {}

    if not particle_df.empty and "diameter_eq_um" in particle_df:
        diameters = particle_df["diameter_eq_um"].dropna().to_numpy(dtype=float)
        diameters = diameters[diameters > 0]
        if diameters.size > 0:
            pooled_fit = fit_lognormal(diameters)
            fig_paths = _save_distribution_plot(diameters, pooled_fit, out_dir)
            figures = {k: str(v) for k, v in fig_paths.items()}
            pooled_summary = {
                "n_particles": int(diameters.size),
                "diameter_mean": float(np.mean(diameters)),
                "diameter_std": float(np.std(diameters)),
                "diameter_median": float(np.median(diameters)),
                "diameter_min": float(np.min(diameters)),
                "diameter_max": float(np.max(diameters)),
                "phase_fraction_mean": float(statistics_df["phase_fraction"].mean()) if not statistics_df.empty else np.nan,
                "phase_fraction_std": float(statistics_df["phase_fraction"].std()) if not statistics_df.empty else np.nan,
            }

    three_d: dict[str, Any] = {"enabled": run_3d, "success": False, "message": "3D not requested."}
    if run_3d and not statistics_df.empty and not particle_df.empty:
        try:
            params = _compute_pooled_graphite_inputs(statistics_df, particle_df)
            engine = (three_d_engine or "matlab").strip().lower()
            if engine == "matlab":
                source_dir = Path(matlab_source_dir) if matlab_source_dir else DEFAULT_MATLAB_SOURCE_DIR
                three_d = _run_matlab_graphite_only(
                    out_dir=out_dir,
                    graphite_vf=params["graphite_vf"],
                    graphite_aspect_ratio=params["graphite_aspect_ratio"],
                    mu_log=params["mu_log"],
                    sigma_log=params["sigma_log"],
                    target_graphite_particles=target_graphite_particles,
                    voxel_size_um=voxel_size_um,
                    matlab_source_dir=source_dir,
                    matlab_iterations=matlab_iterations,
                )
                three_d["pooled_inputs"] = params

                if (not three_d.get("success")) and fallback_to_python_3d:
                    volume, p3d_df, summary3d = _generate_graphite_volume(
                        graphite_vf=params["graphite_vf"],
                        graphite_aspect_ratio=params["graphite_aspect_ratio"],
                        mu_log=params["mu_log"],
                        sigma_log=params["sigma_log"],
                        target_graphite_particles=target_graphite_particles,
                        voxel_size_um=voxel_size_um,
                        max_particles=max_3d_particles,
                        seed=random_seed,
                    )
                    three_d_paths = _save_3d_artifacts(out_dir, volume, p3d_df, summary3d)
                    three_d = {
                        "enabled": True,
                        "success": True,
                        "engine": "python_fallback",
                        "summary": summary3d,
                        "paths": three_d_paths,
                        "volume": volume,
                        "particles_df": p3d_df,
                        "pooled_inputs": params,
                        "message": "MATLAB run unavailable/failed; used Python fallback generator.",
                    }
            else:
                volume, p3d_df, summary3d = _generate_graphite_volume(
                    graphite_vf=params["graphite_vf"],
                    graphite_aspect_ratio=params["graphite_aspect_ratio"],
                    mu_log=params["mu_log"],
                    sigma_log=params["sigma_log"],
                    target_graphite_particles=target_graphite_particles,
                    voxel_size_um=voxel_size_um,
                    max_particles=max_3d_particles,
                    seed=random_seed,
                )
                three_d_paths = _save_3d_artifacts(out_dir, volume, p3d_df, summary3d)
                three_d = {
                    "enabled": True,
                    "success": True,
                    "engine": "python",
                    "summary": summary3d,
                    "paths": three_d_paths,
                    "volume": volume,
                    "particles_df": p3d_df,
                    "pooled_inputs": params,
                }
        except Exception as exc:  # noqa: BLE001
            three_d = {
                "enabled": True,
                "success": False,
                "message": str(exc),
            }

    return {
        "calibration_df": calibration_df,
        "statistics_df": statistics_df,
        "particle_df": particle_df,
        "masks": masks,
        "figures": figures,
        "pooled_fit": pooled_fit,
        "pooled_summary": pooled_summary,
        "errors": errors,
        "output_dir": str(out_dir),
        "per_image_results": results,
        "three_d": three_d,
    }
