# denda-njvid-flight-tracker

> This project uses publicly available insect flight videos from Professor Mitsunori Denda's NJVID collection to explore computational extraction and nonlinear modeling of wing kinematics. The work is intended as an initial demonstration of how computer vision and dynamical systems modeling can support biomechanical research.

## Overview

A clean, modular Python/OpenCV pipeline for extracting wing motion kinematics from high-speed insect flight video. The system tracks insect body centroid and wing tip positions frame-by-frame, computes angular displacement, velocity, acceleration, and wingbeat frequency from real video data — not synthetic sinusoidal approximations.

### What This Project Does

1. **Video Loading** — Loads insect flight video with manual region-of-interest selection for stability.
2. **Body Tracking** — Detects insect body centroid using adaptive thresholding and contour analysis.
3. **Wing Tracking** — Tracks wing tip position using Lucas-Kanade optical flow.
4. **Angular Kinematics** — Computes angular displacement of the wing relative to the body centroid.
5. **Signal Processing** — Savitzky-Golay smoothing, central difference differentiation.
6. **Frequency Analysis** — Estimates wingbeat frequency via FFT and peak detection.
7. **Validation** — Compares automated tracking against manually annotated frames (mean absolute pixel error).
8. **3D Wing Model** — Parametric 3D wing geometry generation for morphological visualization.
9. **Export** — Outputs time-series CSV and publication-quality plots.

### Why This Matters for Biomechanics

Traditional studies of insect flight often approximate wing motion as sinusoidal. Real wing kinematics are more complex — involving nonlinear restoring forces, asymmetric strokes, and coupling with body dynamics. Extracting real kinematic data from video provides the foundation for more accurate dynamical models (see companion project: `nonlinear-wing-model-julia`).

## Data Source

Videos from the **Mitsunori Denda Insect Flight Collection** on NJVID:
https://www.njvid.net/showcollection.php?pid=njcore:16509

Primary species: **Morpho peleides** (`njcore:16554`)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Place video in data/raw/
# Download from NJVID and save as: data/raw/morpho_peleides.mp4
```

## Usage

### Full Pipeline (Interactive)
```bash
python run_tracker.py --video data/raw/morpho_peleides.mp4 --species "Morpho peleides"
```

### With Synthetic Demo Data (No Video Required)
```bash
python run_tracker.py --demo
```

### Generate 3D Wing Model
```bash
python -m src.wing_model --species "Morpho peleides" --output output/models/
```

### Run Tests
```bash
python -m pytest tests/ -v
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/tracking_demo.ipynb
```

## Output

- `output/csv/` — Time-series kinematics (angle, velocity, acceleration)
- `output/plots/` — Trajectory, angle vs time, velocity, FFT spectrum
- `output/models/` — STL 3D wing models
- `output/validation/` — Manual annotation comparison

## Project Structure

```
denda-njvid-flight-tracker/
├── README.md
├── requirements.txt
├── run_tracker.py              # Main entry point
├── src/
│   ├── __init__.py
│   ├── tracker.py              # Video loading, ROI, centroid, optical flow
│   ├── analysis.py             # Smoothing, differentiation, FFT
│   ├── export.py               # CSV export with metadata
│   ├── visualization.py        # Publication-quality plots
│   ├── validate.py             # Manual annotation + MAE
│   └── wing_model.py           # Parametric 3D wing geometry
├── tests/
│   ├── __init__.py
│   └── test_analysis.py        # Synthetic signal validation
├── notebooks/
│   └── tracking_demo.ipynb     # Full pipeline walkthrough
├── data/
│   └── raw/                    # Place NJVID videos here
└── output/
    ├── csv/
    ├── plots/
    └── models/
```

## Citation

If referencing this work, please cite the original video source:

> Denda, M. (2004–2006). Insect Flight Video Collection. NJVID Digital Media Repository,
> Rutgers, The State University of New Jersey.
> https://www.njvid.net/showcollection.php?pid=njcore:16509
