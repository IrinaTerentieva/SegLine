import os
import math
import logging
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiLineString, GeometryCollection
from shapely.ops import split, linemerge
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf


# -----------------------------
# Geometry helper functions
# -----------------------------
def determine_orientation(geometry):
    """
    Determine whether a segment is east-west or north-south based on its bounding box.
    """
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    x_diff = bounds[2] - bounds[0]
    y_diff = bounds[3] - bounds[1]
    orientation = "east-west" if x_diff > y_diff else "north-south"
    return orientation


def get_edge_points(polygon, precision=3):
    """
    Extract all edge points from a polygon's exterior.
    """
    if polygon.is_empty or not polygon.is_valid:
        return set()
    edge_coords = polygon.exterior.coords
    edge_points = {(round(coord[0], precision), round(coord[1], precision)) for coord in edge_coords}
    return edge_points


def sort_segments_and_find_pairs(gdf):
    """
    Sort segments by orientation, assign sides, and find pairs based on shared edge points.
    Returns a GeoDataFrame with a new "plot_id" field.
    """

    def process_unique_id(subset):
        orientation = determine_orientation(subset.geometry.iloc[0])
        subset["centroid_x"] = subset.geometry.centroid.x
        subset["centroid_y"] = subset.geometry.centroid.y
        subset["edge_points"] = subset.geometry.apply(get_edge_points)
        # If PartID is not present, use the index.
        if "PartID" not in subset.columns:
            subset = subset.reset_index().rename(columns={"index": "PartID"})
        else:
            subset = subset.sort_values("PartID")
        subset["side"] = 1  # Default side 1.
        half_rows = len(subset) // 2
        subset.iloc[half_rows:, subset.columns.get_loc("side")] = 0  # Second half: side 0.
        subset["plot_id"] = -1
        segment_id = 0
        side_0 = subset[subset["side"] == 0]
        side_1 = subset[subset["side"] == 1]
        for idx_0, row_0 in side_0.iterrows():
            if row_0.geometry.area < 7:
                continue
            for idx_1, row_1 in side_1.iterrows():
                if row_1.geometry.area < 7:
                    continue
                shared_points = row_0["edge_points"].intersection(row_1["edge_points"])
                if len(shared_points) >= 2:  # Found a pair.
                    subset.at[idx_0, "plot_id"] = segment_id
                    subset.at[idx_1, "plot_id"] = segment_id
                    segment_id += 1
        return subset

    return gdf.groupby("UniqueID", group_keys=False).apply(process_unique_id)


def update_path_with_suffix(input_path: str, suffix: str) -> str:
    """
    Update the input path to include a suffix before the extension.
    """
    if input_path.startswith("file://"):
        input_path = input_path[7:]
    dirname = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    if filename.endswith(".shp"):
        updated_filename = filename.replace(".shp", f"{suffix}.gpkg")
    else:
        updated_filename = filename.replace(".gpkg", f"{suffix}.gpkg")
    return os.path.join(dirname, updated_filename)


# ---------------------------
# Main processing function using Hydra
# ---------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up logging (default INFO level if not specified)
    log_level = cfg.get("logging", {"level": "INFO"}).get("level", "INFO")
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Get spacing from config (used in filename from split_to_plots)
    spacing = int(cfg.split_to_plots.perpendicular_spacing)

    # Build input path from split_to_plots output
    # This should be: {base}_footprint_ID_segments{spacing}m.gpkg
    footprint_base = update_path_with_suffix(cfg.dataset.ground_footprint, "_footprint_ID")
    input_path = footprint_base.replace(".gpkg", f"_segments{spacing}m.gpkg")

    # Build output path
    # This should be: {base}_footprint_ID_segments{spacing}m_sides.gpkg
    output_path = input_path.replace(".gpkg", "_sides.gpkg")

    logging.info(f"Input from split_to_plots: {input_path}")
    logging.info(f"Output for split_to_side: {output_path}")
    logging.info(f"Perpendicular spacing: {spacing}m")

    # Check if input file exists
    if not os.path.exists(input_path):
        logging.error(f"Input file does not exist: {input_path}")
        logging.error("Make sure split_to_plots has been run first!")
        return

    # Read input GeoDataFrame
    try:
        gdf = gpd.read_file(input_path)
        logging.info(f"Read {len(gdf)} segments from {input_path}")
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        return

    # Process segments to sort and find pairs
    paired_gdf = sort_segments_and_find_pairs(gdf)
    logging.info("Finished processing segments for pairing.")

    # Optionally filter small polygons using min_area from configuration
    min_area = cfg.split_to_side.get("min_area", 5)
    initial_count = len(paired_gdf)
    paired_gdf = paired_gdf[paired_gdf.geometry.area >= min_area]
    filtered_count = initial_count - len(paired_gdf)
    logging.info(f"Filtered {filtered_count} small segments (< {min_area} m²)")
    logging.info(f"After filtering, {len(paired_gdf)} segments remain.")

    # Log statistics about sides and plots
    side_0_count = len(paired_gdf[paired_gdf['side'] == 0])
    side_1_count = len(paired_gdf[paired_gdf['side'] == 1])
    paired_count = len(paired_gdf[paired_gdf['plot_id'] >= 0])
    unpaired_count = len(paired_gdf[paired_gdf['plot_id'] == -1])

    logging.info(f"Side 0: {side_0_count} segments")
    logging.info(f"Side 1: {side_1_count} segments")
    logging.info(f"Paired segments: {paired_count}")
    logging.info(f"Unpaired segments: {unpaired_count}")

    # Save the paired segments GeoDataFrame
    paired_gdf.to_file(output_path, driver="GPKG")
    logging.info(f"Paired segments saved to: {output_path}")
    logging.info("Split to sides complete!")


if __name__ == "__main__":
    main()