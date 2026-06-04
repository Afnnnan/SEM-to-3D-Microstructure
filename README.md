# SEM-to-3D Microstructure

> Automated 2D SEM image segmentation, scale-bar calibration, and 3D graphite microstructure generation for cast iron characterization.

This project provides a **Streamlit-based UI** that takes 2D SEM micrographs of cast iron samples, performs automated segmentation and calibration, and generates a 3D graphite microstructure via ellipsoid packing.

---

## Features

- **Batch upload** of SEM images (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`)
- **Automated scale-bar detection** via OCR (Tesseract), with manual override support
- **Graphite particle segmentation** with configurable diameter and circularity thresholds
- **Pooled lognormal fitting** across multiple images
- **3D ellipsoid packing** — supports both MATLAB (Kasemer et al.) and a built-in Python fallback engine
- **Interactive 3D visualization** using Plotly (ellipsoid meshes + voxel boundary views)
- **Downloadable output archive** (CSV tables, masks, fit plots, 3D volumes)

---

## Important Note — MATLAB Engine

The MATLAB-based 3D packing code (Kasemer et al.) is **not included** in this repository. If you run the app without a local MATLAB installation and the packing source code, the pipeline will **automatically fall back to the built-in Python 3D packer**, which produces comparable results.

If you do have access to the MATLAB code, configure it via a `.env` file (see [`.env.example`](streamlit_ui/.env.example)).

---

## Quick Start

```bash
# Clone
git clone https://github.com/Afnannan/SEM-to-3D-Microstructure.git
cd SEM-to-3D-Microstructure/streamlit_ui

# Install dependencies
pip install -r requirements_streamlit.txt

# (Optional) Tesseract for OCR-based scale detection
# macOS: brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr

# Run
streamlit run app.py
```

---

## Configuration

### Environment Variables

Copy the example env file and edit it for your machine:

```bash
cp streamlit_ui/.env.example streamlit_ui/.env
```

| Variable | Description |
|---|---|
| `MATLAB_EXE` | Full path to the MATLAB executable |
| `MATLAB_SOURCE_DIR` | Path to the MATLAB graphite-only packing codebase |
| `TESSERACT_CMD` | *(Optional)* Path to the Tesseract binary if not on `PATH` |

### Sidebar Options

The Streamlit sidebar exposes all tunable parameters:

| Section | Parameters |
|---|---|
| **2D Processing** | Metadata strip ratio, OCR fallback scale, flake-case preset, min diameter, min circularity |
| **3D Generation** | Engine (MATLAB/Python), target particles, voxel size, max particles, random seed, MATLAB iterations |
| **Display** | Per-image intermediates toggle, max scatter points, max ellipsoids, mesh resolution |

---

## Outputs

Each pipeline run writes to `streamlit_outputs/<timestamp>/` and produces a downloadable `.zip` containing:

- `calibration_results.csv` — per-image scale calibration
- `summary_statistics.csv` — segmentation statistics
- `particle_data.csv` — individual particle measurements
- `<image>_mask.tif` — binary segmentation masks
- `pooled_size_distribution.png` / `lognormal_fit.png` — distribution plots
- `graphite_only_inputs.txt` — inputs fed to the 3D engine
- `graphite_particles_3d.csv` — packed ellipsoid centers & radii
- `graphite_volume.npy` — voxelized 3D volume
- `graphite_volume_mid_slices.png` — XY/XZ/YZ mid-slice previews

---

## Additional Information

For a detailed overview of the methodology, pipeline architecture, and results, refer to the **[Final Presentation](FinalPresentation.pdf)** included in this repository.

---

## Project Structure

```
.
├── README.md
├── FinalPresentation.pdf
└── streamlit_ui/
    ├── app.py                      # Streamlit frontend
    ├── pipeline.py                 # Core segmentation, calibration & 3D generation
    ├── requirements_streamlit.txt  # Python dependencies
    ├── .env.example                # Environment variable template
    └── README.md                   # Original app-level docs
```

---

## License

This project was developed as part of the **MM226** course. Please contact the author before reuse.
