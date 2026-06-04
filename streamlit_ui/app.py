from __future__ import annotations

import os
from pathlib import Path
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_file(BASE_DIR / ".env")

from pipeline import run_batch


def _default_matlab_source_dir() -> str:
    env_path = os.environ.get("MATLAB_SOURCE_DIR", "").strip()
    if env_path:
        return str(Path(env_path).expanduser())
    return str((BASE_DIR / "matlab_source").resolve())


DEFAULT_MATLAB_SOURCE_DIR = _default_matlab_source_dir()


def _read_bytes_if_exists(path_value: str | None) -> bytes | None:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.exists() or not p.is_file():
        return None
    return p.read_bytes()


def _image_payload(value: object, inline_value: bytes | None) -> object | None:
    if inline_value is not None:
        return inline_value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        p = Path(value)
        if p.exists() and p.is_file():
            return value
        return None
    return None


def _build_output_zip(output_dir: Path) -> Path:
    zip_path = output_dir / "streamlit_outputs.zip"
    files_to_zip = [path for path in output_dir.rglob("*") if path.is_file() and path != zip_path]
    if not files_to_zip:
        if zip_path.exists() and zip_path.stat().st_size > 22:
            return zip_path
        raise RuntimeError(
            f"No output files available to zip in {output_dir}. "
            "Rerun the pipeline to regenerate outputs."
        )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files_to_zip:
            zf.write(path, arcname=str(path.relative_to(output_dir)))
    return zip_path


def _cleanup_output_dir_keep_zip(output_dir: Path, zip_path: Path) -> None:
    for path in output_dir.iterdir():
        if path == zip_path:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _apply_graphite_case_preset() -> None:
    if st.session_state.get("flake_case_preset", False):
        st.session_state["min_diameter_um_value"] = 0.5
        st.session_state["min_circularity_value"] = 0.0
    else:
        st.session_state["min_diameter_um_value"] = 5.0
        st.session_state["min_circularity_value"] = 0.2


if "min_diameter_um_value" not in st.session_state:
    st.session_state["min_diameter_um_value"] = 5.0
if "min_circularity_value" not in st.session_state:
    st.session_state["min_circularity_value"] = 0.2
if "flake_case_preset" not in st.session_state:
    st.session_state["flake_case_preset"] = False
if "upload_nonce" not in st.session_state:
    st.session_state["upload_nonce"] = 0

st.set_page_config(page_title="2D to 3D", layout="wide")

st.title("2D SiMo to 3D Graphite Microstructure")
st.caption(
    "Uploads 2D SEM images, runs segmentation-calibration pipeline, "
    "then generates a graphite-only 3D microstructure inspired by graphite_only_main.m."
)

uploaded_files = st.file_uploader(
    "Upload one or more SEM images (.tif/.tiff/.png/.jpg/.jpeg)",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"uploaded_files_{st.session_state['upload_nonce']}",
    help="Upload SEM images for batch processing. Duplicate filenames are ignored after the first one.",
)

top_actions_col1, top_actions_col2 = st.columns([1, 5])
with top_actions_col1:
    if st.button(
        "Clear All Images",
        disabled=not uploaded_files,
        help="Flush all uploaded images from the uploader without refreshing the whole page.",
    ):
        st.session_state["upload_nonce"] += 1
        st.session_state.pop("latest_batch", None)
        st.session_state.pop("interactive_3d_started_for", None)
        st.rerun()

deduped_uploaded_files = []
duplicate_names: list[str] = []
seen_names: set[str] = set()
for f in uploaded_files or []:
    if f.name in seen_names:
        duplicate_names.append(f.name)
        continue
    seen_names.add(f.name)
    deduped_uploaded_files.append(f)

uploaded_files = deduped_uploaded_files
if duplicate_names:
    skipped = ", ".join(sorted(set(duplicate_names)))
    st.warning(f"Duplicate file(s) ignored (kept first occurrence): {skipped}")

with st.sidebar:
    st.header("2D Processing")
    strip_ratio = st.slider(
        "Metadata strip ratio (bottom fraction)",
        min_value=0.02,
        max_value=0.20,
        value=0.073,
        step=0.001,
        format="%.3f",
        help="Fraction of image height treated as metadata strip at the bottom (used for scale-bar OCR).",
    )
    fallback_um = st.number_input(
        "OCR fallback scale (um)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
        help="Scale value used when OCR cannot reliably read the scale bar text.",
    )
    st.toggle(
        "Flake case preset",
        key="flake_case_preset",
        on_change=_apply_graphite_case_preset,
        help="On: uses flake-like minimum diameter/circularity. Off: restores nodule defaults.",
    )
    min_diameter_um = st.number_input(
        "Min graphite diameter (um)",
        min_value=0.5,
        max_value=30.0,
        step=0.5,
        key="min_diameter_um_value",
        help="Minimum equivalent graphite particle diameter retained during 2D segmentation.",
    )
    min_circularity = st.slider(
        "Min circularity (simple branch)",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="min_circularity_value",
        help="Circularity threshold for particle filtering in the simple segmentation branch.",
    )

    st.header("3D Generation")
    run_3d = st.checkbox(
        "Generate 3D graphite microstructure",
        value=True,
        help="If off, only 2D analysis is run.",
    )
    three_d_engine = st.selectbox(
        "3D engine",
        options=["matlab", "python"],
        index=0,
        help="matlab: run Kasemer MATLAB code with pooled stats. python: use local fallback packer.",
    )
    matlab_source_dir = st.text_input(
        "MATLAB source dir",
        value=DEFAULT_MATLAB_SOURCE_DIR,
        help="Path to the MATLAB graphite-only packing codebase.",
    )
    matlab_iterations = st.number_input(
        "MATLAB iterations",
        min_value=1,
        max_value=25,
        value=1,
        step=1,
        help="Number of MATLAB packing iterations to run.",
    )
    fallback_to_python_3d = st.checkbox(
        "Fallback to Python if MATLAB fails",
        value=True,
        help="Automatically run the Python packer if MATLAB execution fails.",
    )
    target_graphite_particles = st.number_input(
        "Target graphite particles",
        min_value=30,
        max_value=1000,
        value=200,
        step=10,
        help="Target number of graphite particles for 3D packing.",
    )
    voxel_size_um = st.number_input(
        "Voxel size (um)",
        min_value=0.25,
        max_value=10.0,
        value=2.0,
        step=0.25,
        help="Voxel edge length used for volume discretization.",
    )
    max_3d_particles = st.number_input(
        "Max particles to pack",
        min_value=50,
        max_value=1500,
        value=350,
        step=10,
        help="Hard cap on packed particles in Python fallback mode.",
    )
    random_seed = st.number_input(
        "Random seed (0 = random)",
        min_value=0,
        max_value=999999,
        value=0,
        step=1,
        help="Set a non-zero value for reproducible runs.",
    )

    st.header("Display")
    show_intermediates = st.checkbox(
        "Show per-image intermediates",
        value=True,
        help="Show original/overlay/mask previews and key stats for each image.",
    )
    max_scatter_points = st.number_input(
        "Max 3D scatter points",
        min_value=2000,
        max_value=100000,
        value=100000,
        step=1000,
        help="Upper limit of plotted points in voxel-boundary fallback view.",
    )
    max_ellipsoids = st.number_input(
        "Max 3D ellipsoids",
        min_value=10,
        max_value=300,
        value=300,
        step=10,
        help="Maximum ellipsoids rendered in the interactive surface view.",
    )
    mesh_resolution = st.slider(
        "Ellipsoid mesh resolution",
        min_value=12,
        max_value=40,
        value=40,
        step=2,
        help="Surface smoothness for each ellipsoid; higher is smoother but heavier.",
    )


manual_overrides: dict[str, dict[str, float | int | None]] = {}
if uploaded_files:
    with st.expander("Manual override (optional)", expanded=False):
        st.caption("Use this to bypass scale detection with a direct um/px value and optionally crop from the bottom.")
        for f in uploaded_files:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                use_override = st.checkbox(
                    f"Override calibration for {f.name}",
                    value=False,
                    key=f"override_toggle_{f.name}",
                    help="Enable direct microns-per-pixel input for this image instead of scale detection.",
                )
            with col2:
                override_mpp = st.number_input(
                    f"um/px ({f.name})",
                    min_value=0.000001,
                    max_value=1000.0,
                    value=1.0,
                    step=0.0001,
                    format="%.6f",
                    disabled=not use_override,
                    key=f"override_mpp_{f.name}",
                    help="Direct micrometers-per-pixel override for this image.",
                )
            with col3:
                exclude_bottom_px = st.number_input(
                    f"Bottom px ({f.name})",
                    min_value=0,
                    max_value=5000,
                    value=0,
                    step=1,
                    key=f"exclude_bottom_px_{f.name}",
                    help="Pixels to remove from the bottom before segmentation.",
                )
            manual_overrides[f.name] = {
                "manual_scale_um": None,
                "manual_microns_per_pixel": float(override_mpp) if use_override else None,
                "exclude_bottom_px": int(exclude_bottom_px),
            }


run = st.button(
    "Run Pipeline",
    type="primary",
    disabled=not uploaded_files,
    help="Run 2D analysis and optional 3D generation for all uploaded images.",
)

if run_3d and three_d_engine == "matlab":
    matlab_path = Path(matlab_source_dir).expanduser()
    matlab_exe_env = os.environ.get("MATLAB_EXE", "").strip()
    matlab_cli_found = bool((matlab_exe_env and Path(matlab_exe_env).exists()) or shutil.which("matlab"))
    if not matlab_path.exists():
        st.warning(
            "MATLAB source path not found. Set `MATLAB source dir` in the sidebar or export "
            "`MATLAB_SOURCE_DIR` before launching Streamlit."
        )
    if not matlab_cli_found:
        st.warning(
            "MATLAB CLI not found. Install MATLAB and ensure `matlab` is on PATH, or set `MATLAB_EXE` "
            "to the full executable path."
        )




def render_3d_views(volume: np.ndarray, max_points: int, voxel_size_um: float) -> None:
    vol = volume.astype(bool)
    if vol.size == 0 or not np.any(vol):
        st.info("No occupied voxels to display for 3D volume.")
        return

    zmid, ymid, xmid = np.array(vol.shape) // 2
    fig_slice, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(vol[zmid, :, :], cmap="gray")
    axes[0].set_title("XY mid-slice")
    axes[1].imshow(vol[:, ymid, :], cmap="gray")
    axes[1].set_title("XZ mid-slice")
    axes[2].imshow(vol[:, :, xmid], cmap="gray")
    axes[2].set_title("YZ mid-slice")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig_slice, clear_figure=True)

    # Boundary voxels for cleaner scatter.
    eroded = np.zeros_like(vol)
    if min(vol.shape) > 2:
        eroded[1:-1, 1:-1, 1:-1] = (
            vol[1:-1, 1:-1, 1:-1]
            & vol[:-2, 1:-1, 1:-1]
            & vol[2:, 1:-1, 1:-1]
            & vol[1:-1, :-2, 1:-1]
            & vol[1:-1, 2:, 1:-1]
            & vol[1:-1, 1:-1, :-2]
            & vol[1:-1, 1:-1, 2:]
        )
    boundary = vol & ~eroded

    zz, yy, xx = np.where(boundary)
    n = len(xx)
    if n == 0:
        zz, yy, xx = np.where(vol)
        n = len(xx)

    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        xx, yy, zz = xx[idx], yy[idx], zz[idx]

    marker_size = 2 if n <= 12000 else 1
    x_um = xx.astype(float) * float(voxel_size_um)
    y_um = yy.astype(float) * float(voxel_size_um)
    z_um = zz.astype(float) * float(voxel_size_um)

    fig_3d = go.Figure(
        data=[
            go.Scatter3d(
                x=x_um,
                y=y_um,
                z=z_um,
                mode="markers",
                marker={
                    "size": marker_size,
                    "opacity": 0.65,
                    "color": zz.astype(float),
                    "colorscale": "Turbo",
                    "showscale": False,
                },
                name="Graphite boundary",
            )
        ]
    )
    fig_3d.update_layout(
        title="Interactive 3D Graphite Voxel Boundary (um)",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        scene={
            "xaxis_title": "X (um)",
            "yaxis_title": "Y (um)",
            "zaxis_title": "Z (um)",
            "aspectmode": "data",
            "dragmode": "orbit",
        },
    )
    st.plotly_chart(fig_3d, width="stretch")


def _rotation_matrix_zxz(phi1: float, Phi: float, phi2: float) -> np.ndarray:
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    rz1 = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    rz2 = np.array([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
    return rz1 @ rx @ rz2


def _unit_sphere_grid(mesh_resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, int(mesh_resolution), endpoint=True)
    v = np.linspace(0.0, np.pi, int(max(8, mesh_resolution // 2)), endpoint=True)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    su = np.cos(uu) * np.sin(vv)
    sv = np.sin(uu) * np.sin(vv)
    sw = np.cos(vv)
    return su, sv, sw


def render_ellipsoid_meshes(
    particles_df: pd.DataFrame,
    max_ellipsoids: int,
    mesh_resolution: int,
) -> bool:
    if particles_df is None or particles_df.empty:
        return False

    df = particles_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    has_um = {"cx_um", "cy_um", "cz_um", "rx_um", "ry_um", "rz_um"}.issubset(df.columns)
    has_vox = {"cx_vox", "cy_vox", "cz_vox", "rx_vox", "ry_vox", "rz_vox"}.issubset(df.columns)
    if not has_um and not has_vox:
        return False

    if has_um:
        cx_col, cy_col, cz_col = "cx_um", "cy_um", "cz_um"
        rx_col, ry_col, rz_col = "rx_um", "ry_um", "rz_um"
        unit_label = "um"
    else:
        cx_col, cy_col, cz_col = "cx_vox", "cy_vox", "cz_vox"
        rx_col, ry_col, rz_col = "rx_vox", "ry_vox", "rz_vox"
        unit_label = "vox"

    valid = (
        np.isfinite(df[cx_col].to_numpy(dtype=float))
        & np.isfinite(df[cy_col].to_numpy(dtype=float))
        & np.isfinite(df[cz_col].to_numpy(dtype=float))
        & np.isfinite(df[rx_col].to_numpy(dtype=float))
        & np.isfinite(df[ry_col].to_numpy(dtype=float))
        & np.isfinite(df[rz_col].to_numpy(dtype=float))
    )
    df = df.loc[valid].copy()
    if df.empty:
        return False

    df["ellipsoid_volume"] = (
        4.0
        * np.pi
        / 3.0
        * df[rx_col].astype(float)
        * df[ry_col].astype(float)
        * df[rz_col].astype(float)
    )
    df = df.sort_values("ellipsoid_volume", ascending=False).head(int(max_ellipsoids))
    if len(df) < len(particles_df):
        st.caption(f"Rendering top {len(df)} ellipsoids by volume for performance.")

    su, sv, sw = _unit_sphere_grid(int(mesh_resolution))

    fig = go.Figure()
    color_values = np.linspace(0.2, 1.0, max(2, len(df)))
    use_orientation = {"phi1_rad", "Phi_rad", "phi2_rad"}.issubset(df.columns)

    for i, (_, row) in enumerate(df.iterrows()):
        rx, ry, rz = float(row[rx_col]), float(row[ry_col]), float(row[rz_col])
        cx, cy, cz = float(row[cx_col]), float(row[cy_col]), float(row[cz_col])

        x0 = rx * su
        y0 = ry * sv
        z0 = rz * sw

        if use_orientation:
            rot = _rotation_matrix_zxz(float(row["phi1_rad"]), float(row["Phi_rad"]), float(row["phi2_rad"]))
            pts = np.vstack([x0.ravel(), y0.ravel(), z0.ravel()])
            rot_pts = rot @ pts
            x = rot_pts[0].reshape(x0.shape) + cx
            y = rot_pts[1].reshape(y0.shape) + cy
            z = rot_pts[2].reshape(z0.shape) + cz
        else:
            x = x0 + cx
            y = y0 + cy
            z = z0 + cz

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=np.full_like(x, color_values[i]),
                colorscale="Turbo",
                cmin=0.0,
                cmax=1.0,
                showscale=False,
                opacity=0.9,
                hovertemplate=(
                    f"Particle #{i + 1}<br>"
                    f"Center ({unit_label}): ({cx:.2f}, {cy:.2f}, {cz:.2f})<br>"
                    f"Radii ({unit_label}): ({rx:.2f}, {ry:.2f}, {rz:.2f})<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Interactive 3D Ellipsoid Packing ({unit_label})",
        height=860,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis_title": f"X ({unit_label})",
            "yaxis_title": f"Y ({unit_label})",
            "zaxis_title": f"Z ({unit_label})",
            "aspectmode": "data",
            "dragmode": "orbit",
            "camera": {"projection": {"type": "perspective"}},
        },
    )
    st.plotly_chart(fig, width="stretch")
    return True


if run and uploaded_files:
    seed = None if int(random_seed) == 0 else int(random_seed)

    with st.spinner("Running 2D analysis and 3D generation..."):
        batch = run_batch(
            files=uploaded_files,
            strip_ratio=float(strip_ratio),
            fallback_um=float(fallback_um),
            manual_overrides=manual_overrides,
            min_diameter_um=float(min_diameter_um),
            min_circularity=float(min_circularity),
            run_3d=run_3d,
            target_graphite_particles=int(target_graphite_particles),
            voxel_size_um=float(voxel_size_um),
            max_3d_particles=int(max_3d_particles),
            random_seed=seed,
            three_d_engine=three_d_engine,
            matlab_source_dir=matlab_source_dir,
            matlab_iterations=int(matlab_iterations),
            fallback_to_python_3d=fallback_to_python_3d,
        )
    inline_assets: dict[str, bytes] = {}
    figs_for_cache = batch.get("figures", {})
    pooled_bytes = _read_bytes_if_exists(figs_for_cache.get("pooled_size_distribution"))
    lognormal_bytes = _read_bytes_if_exists(figs_for_cache.get("lognormal_fit"))
    if pooled_bytes is not None:
        inline_assets["pooled_size_distribution"] = pooled_bytes
    if lognormal_bytes is not None:
        inline_assets["lognormal_fit"] = lognormal_bytes

    three_d_cache = batch.get("three_d", {})
    packing_path_cache = (three_d_cache.get("paths") or {}).get("packing_png")
    packing_bytes = _read_bytes_if_exists(packing_path_cache)
    if packing_bytes is not None:
        inline_assets["packing_png"] = packing_bytes
    batch["_inline_assets"] = inline_assets

    st.session_state["latest_batch"] = batch
    st.session_state["interactive_3d_started_for"] = None

batch = st.session_state.get("latest_batch")

if batch is not None:

    output_dir = Path(batch["output_dir"])
    st.success(f"Done. Outputs saved to: {output_dir}")

    if batch["errors"]:
        st.warning("Some files failed during 2D processing.")
        st.dataframe(pd.DataFrame(batch["errors"]), width="stretch")

    calibration_df: pd.DataFrame = batch["calibration_df"]
    statistics_df: pd.DataFrame = batch["statistics_df"]
    particle_df: pd.DataFrame = batch["particle_df"]

    st.subheader("2D Batch Tables")
    t1, t2, t3 = st.tabs(["Calibration", "Summary Stats", "Particle Data"])
    with t1:
        st.dataframe(calibration_df, width="stretch")
    with t2:
        st.dataframe(statistics_df, width="stretch")
    with t3:
        st.dataframe(particle_df, width="stretch")

    st.subheader("Pooled 2D Distribution")
    pooled_summary = batch.get("pooled_summary", {})
    pooled_fit = batch.get("pooled_fit", {})

    if pooled_summary:
        st.json(pooled_summary)
    if pooled_fit.get("success"):
        st.write(
            {
                "mu": pooled_fit["mu"],
                "sigma": pooled_fit["sigma"],
                "ks_stat": pooled_fit["ks_stat"],
                "p_value": pooled_fit["p_value"],
            }
        )
    else:
        st.info(pooled_fit.get("message", "Pooled fit unavailable."))

    figs = batch.get("figures", {})
    inline_assets = batch.get("_inline_assets", {})
    pooled_img = _image_payload(figs.get("pooled_size_distribution"), inline_assets.get("pooled_size_distribution"))
    lognormal_img = _image_payload(figs.get("lognormal_fit"), inline_assets.get("lognormal_fit"))

    if pooled_img and lognormal_img:
        tp1, tp2 = st.tabs(["Pooled size distribution", "Lognormal fit"])
        with tp1:
            st.image(pooled_img, width="stretch")
        with tp2:
            l1, l2, l3 = st.columns([1, 6, 1])
            with l2:
                st.image(lognormal_img, width="stretch")
    else:
        colf1, colf2 = st.columns(2)
        if pooled_img:
            colf1.image(pooled_img, width="stretch")
        if lognormal_img:
            colf2.image(lognormal_img, width="stretch")

    if show_intermediates:
        st.subheader("Per-image Intermediates")
        for result in batch["per_image_results"]:
            with st.expander(result["filename"], expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.image(result["original_image"], caption="Original", clamp=True)
                c2.image(result["overlay"], caption="Overlay", clamp=True)
                c3.image((result["mask"].astype(np.uint8) * 255), caption="Mask", clamp=True)
                st.write(result["calibration"])
                st.write({
                    k: result["statistics"][k]
                    for k in [
                        "strategy",
                        "phase_fraction",
                        "n_particles",
                        "number_density_um2",
                        "diameter_um_mean",
                        "diameter_um_std",
                    ]
                })

    st.subheader("3D Graphite Microstructure")
    three_d = batch.get("three_d", {})
    if not three_d.get("enabled", True):
        st.info("3D generation is disabled from sidebar.")
    elif not three_d.get("success"):
        st.warning(three_d.get("message", "3D generation failed."))
        if three_d.get("paths"):
            st.json(three_d.get("paths"))
    else:
        st.write({"engine": three_d.get("engine", "unknown")})
        if three_d.get("message"):
            st.info(three_d.get("message"))
        st.markdown("**Pooled inputs used for 3D**")
        st.json(three_d.get("pooled_inputs", {}))
        st.markdown("**Packed 3D summary**")
        st.json(three_d.get("summary", {}))
        if three_d.get("paths"):
            with st.expander("3D output paths", expanded=False):
                st.json(three_d.get("paths"))

        packing_png = (three_d.get("paths") or {}).get("packing_png")
        packing_img = _image_payload(packing_png, inline_assets.get("packing_png"))
        if packing_img:
            img_col1, img_col2, img_col3 = st.columns([1, 6, 1])
            with img_col2:
                st.image(
                    packing_img,
                    caption="MATLAB 3D packing view",
                    width=800,
                )

        p3d = three_d.get("particles_df")
        rendered_mesh = False
        if isinstance(p3d, pd.DataFrame) and not p3d.empty:
            current_batch_id = str(output_dir)
            start_key = f"start_interactive_3d_{current_batch_id}"
            if st.button("Start Interactive 3D", key=start_key):
                st.session_state["interactive_3d_started_for"] = current_batch_id

            if st.session_state.get("interactive_3d_started_for") == current_batch_id:
                rendered_mesh = render_ellipsoid_meshes(
                    particles_df=p3d,
                    max_ellipsoids=int(max_ellipsoids),
                    mesh_resolution=int(mesh_resolution),
                )
            else:
                st.info("Click `Start Interactive 3D` to load interactive mesh view.")

        volume = three_d.get("volume")
        if isinstance(volume, np.ndarray):
            with st.expander("Voxel view (fallback/diagnostic)", expanded=not rendered_mesh):
                render_3d_views(volume, int(max_scatter_points), float(voxel_size_um))

        if isinstance(p3d, pd.DataFrame) and not p3d.empty:
            with st.expander("Packed ellipsoid centers/radii", expanded=False):
                st.dataframe(p3d.head(300), width="stretch")

    st.subheader("Download outputs")
    zip_path = _build_output_zip(output_dir)
    _cleanup_output_dir_keep_zip(output_dir, zip_path)
    zip_bytes = zip_path.read_bytes()
    st.download_button(
        "Download all outputs (.zip)",
        data=zip_bytes,
        file_name=f"{output_dir.name}_outputs.zip",
        mime="application/zip",
    )
