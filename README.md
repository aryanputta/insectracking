# InsecTracking: Biomechanical Flight Analysis & 3D Reconstruction

**Author:** Aryan Putta  
**Project:** Computer Vision-Based Analysis of Lepidoptera Kinematics  

## Overview
This repository provides a high-density, automated pipeline for extracting biomechanical flight data from insect videos. It uses advanced computer vision techniques (Lucas-Kanade optical flow, Shi-Tomasi feature detection, and HSL-based skin exclusion) to track anatomical landmarks and dense texture points, which are then used to drive a parametric 3D model.

## Visual Results

### 1. Live Tracking Demo
The system extracts 36 dense keypoints (11 anatomical + 25 features) in real-time. Note how the **skin-exclusion logic** prevents the head keypoint from jumping to the human finger.
![Live Tracking Demo](docs/images/tracking_demo.gif)

### 2. High-Density Tracking Dashboard
The tracker provides a complete kinematic profile of the flight trajectory and multi-point spread.
![Tracking Dashboard](docs/images/tracking_dashboard.png)

### 3. PCA Motion Compression
We use Principal Component Analysis to distill the complex wing motions. The first 3 components capture **98.6% of the total variance**.
![PCA Decomposition](docs/images/pca_decomposition.png)

## Key Features
- **Dense Keypoint Tracker**: Tracks 11 anatomical landmarks and 25+ texture features per frame.
- **Finger Avoidance (Skin Exclusion)**: Uses HSV + YCrCb masking to prevent tracking errors caused by human handling.
- **Kinematic Analysis**: Automatically calculates per-frame displacement, velocity, and wing area dynamics.
- **Parametric 3D Modeling**: Compresses flight motion using PCA (capturing ~98.9% variance) and exports articulated STL meshes.
- **Unified Pipeline**: A single command processes raw video into a full research package.

## Repository Structure
```
denda-njvid-flight-tracker/
├── run_pipeline.py         # Unified entry point (Live Track -> 3D Mesh)
├── multipoint_tracker.py   # High-density computer vision tracker
├── extract_kinematics.py   # Position/Velocity/Area analysis
├── parametric_3d_model.py  # PCA-based 3D reconstruction
├── output/                 # Results (CSVs, Plots, STLs)
└── data/                   # Raw research video
```

## Getting Started
1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execution**:
   Run the unified research pipeline with live visualization:
   ```bash
   python3 run_pipeline.py data/raw/morpho_peleides.mp4 --live
   ```

## Output Data
- **`keypoints_all_frames.csv`**: Raw (x, y) coordinates for all tracked points.
- **`kinematics_per_frame.csv`**: Derived velocity, displacement, and wing area deltas.
- **`3d_simulation/animation/`**: Articulated STL meshes for every frame of the flight sequence.
- **`plots/tracking_dashboard.png`**: Multi-panel visualization of trajectories.

---
*Developed for research in insect flight biomechanics.*
