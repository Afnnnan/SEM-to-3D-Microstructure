# Streamlit UI: 2D to 3D Graphite Microstructure

This app runs segmentation + calibration on uploaded 2D SEM images, then drives a graphite-only 3D generator integrated with your local MATLAB packing code (`graphite_only_main.m`).

## What It Does

- Upload one or more 2D SEM images.
- Run calibration + segmentation + pooled lognormal fitting.
- Compute pooled 3D inputs (`vf`, `aspect_ratio`, `mu_log`, `sigma_log`).
- Feed those pooled stats into the MATLAB packing code when `3D engine = matlab`.
- Display MATLAB-generated 3D output and a voxelized reconstruction in Streamlit.
- Optionally use a Python fallback 3D generator if MATLAB is unavailable.
- Optionally override calibration with a direct `um/px` value and crop pixels from the bottom before segmentation.

## Run

From this folder:

```bash
python -m pip install -r requirements_streamlit.txt
python -m streamlit run app.py
```

If OCR binary is not detected, fallback/manual scale is used.
Optionally set:

```bash
export TESSERACT_CMD="/opt/homebrew/bin/tesseract"
```

For MATLAB engine, ensure MATLAB CLI is available and set paths on each machine.

Option 1: use `.env` (recommended)

```bash
cp .env.example .env
# edit .env values for your machine
```

Option 2: export in shell

```bash
export MATLAB_EXE="/full/path/to/matlab"
export MATLAB_SOURCE_DIR="/full/path/to/matlab/packing/code"
```

If `MATLAB_SOURCE_DIR` is not set, the app defaults to `./matlab_source` relative to this folder.

## Outputs

Each run writes to:

`./streamlit_outputs/<timestamp>/`

The app keeps a ZIP archive of the outputs and removes the uncompressed files afterward.

Typical contents include:

- `calibration_results.csv`
- `summary_statistics.csv`
- `particle_data.csv`
- `<image_name>_mask.tif`
- `pooled_size_distribution.png`
- `lognormal_fit.png`
- `graphite_only_inputs.txt`
- `graphite_only_results_matrix.csv`
- `graphite_particles_3d.csv`
- `graphite_volume.npy`
- `graphite_volume_mid_slices.png`

## Notes

- For paper figures or non-standard image layouts, use the manual override section to enter `um/px` directly and remove a chosen number of pixels from the bottom.
