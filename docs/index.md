# SegLine: Geospatial Segmentation Pipeline

**SegLine** is a modular and reproducible pipeline for geospatial polygon segmentation, designed to create hierarchical splits of land features — from large footprints to fine subplots.

This pipeline is built using Python, GeoPandas, Shapely, and Hydra for configuration management. It supports parallel processing and flexible customization for various geographic datasets.

---

## 🔧 Pipeline Overview

The segmentation process is broken into five clearly defined steps:

1. **Assign Unique IDs**  
   Assigns a consistent `UniqueID` to each polygon and matches it with a corresponding centerline by spatial sampling.

2. **Smooth Centerlines**  
   Identifies sharp angle changes in the centerlines and smooths them to improve splitting quality.

3. **Split to Plots**  
   Uses the centerline and smoothed geometry to split each polygon into segments of a target area (e.g., 100 m²).

4. **Split to Sides**  
   Detects symmetrical side-pairs of segments (east-west or north-south) and assigns `side` and `SegmentID`.

5. **Split to Subplots**  
   Performs a second-level split of each side segment into smaller subplots (e.g., 10 m²), assigning each a `plot_id`.

---

## 📁 Project Structure


---

## 📸 Examples

- ![Footprint + Centerline](examples/1_line_footprint.png)
- ![Smoothed Centerline](examples/2_smooth_centerline.png)
- ![Segmenting to Plots](examples/3_plots.png)
- ![Defining Sides and Plots](examples/4_sides.png)
- ![Segmenting to Subplots](examples/5_subplots.png)

---

## 🚀 Usage

Run the full pipeline using:

```bash
python run_pipeline.py
