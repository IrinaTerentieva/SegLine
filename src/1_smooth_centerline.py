import os
import math
import logging
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import substring, linemerge
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf
import random

# -----------------------------
# Geometry smoothing functions
# -----------------------------
def calculate_angle(p1: Point, p2: Point, p3: Point) -> float:
    """
    Calculate the angle (in degrees) between three points p1, p2, and p3.
    """
    dx1, dy1 = p2.x - p1.x, p2.y - p1.y
    dx2, dy2 = p3.x - p2.x, p3.y - p2.y
    dot_product = dx1 * dx2 + dy1 * dy2
    magnitude1 = math.sqrt(dx1**2 + dy1**2)
    magnitude2 = math.sqrt(dx2**2 + dy2**2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 180  # Treat as straight line if any segment is zero length
    angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
    return math.degrees(angle_rad)

def smooth_within_buffer(line: LineString, bad_point: Point, buffer_distance: float) -> LineString:
    """
    Smooth a section of the line around a bad vertex (within buffer_distance).
    """
    try:
        buffer = bad_point.buffer(buffer_distance)
        line_within_buffer = line.intersection(buffer)
        if isinstance(line_within_buffer, LineString) and not line_within_buffer.is_empty:
            coords = list(line_within_buffer.coords)
            # Interpolate between start and end of the segment to smooth
            smoothed_coords = np.linspace(np.array(coords[0]), np.array(coords[-1]), len(coords))
            smoothed_segment = LineString(smoothed_coords)
            start_proj = line.project(Point(smoothed_segment.coords[0]))
            end_proj = line.project(Point(smoothed_segment.coords[-1]))
            start_part = substring(line, 0, start_proj, normalized=False)
            end_part = substring(line, end_proj, line.length, normalized=False)
            return LineString(list(start_part.coords) + list(smoothed_segment.coords) + list(end_part.coords))
    except Exception as e:
        logging.error(f"Error smoothing buffer: {e}")
    return line

def smooth_centerline(line: LineString, angle_threshold: float = 130, buffer_distance: float = 2) -> LineString:
    """
    Smooth vertices of a LineString if their angles fall outside the acceptable range.
    If the input geometry is a MultiLineString, merge it first.
    """
    if isinstance(line, MultiLineString):
        line = linemerge(line)
    if not isinstance(line, LineString) or len(line.coords) < 3:
        return line
    coords = [Point(coord) for coord in line.coords]
    for i in range(1, len(coords) - 1):
        try:
            angle = calculate_angle(coords[i - 1], coords[i], coords[i + 1])
            if angle < angle_threshold or angle > (360 - angle_threshold):
                line = smooth_within_buffer(line, coords[i], buffer_distance)
        except Exception as e:
            logging.error(f"Error processing angle at index {i}: {e}")
    return line

def process_centerline_line(line: LineString, angle_threshold: float, buffer_distance: float) -> LineString:
    """
    Process a single line for smoothing.
    """
    try:
        return smooth_centerline(line, angle_threshold, buffer_distance)
    except Exception as e:
        logging.error(f"Error processing line: {e}")
        return line

def process_centerlines_worker(line: LineString, angle_threshold: float, buffer_distance: float) -> LineString:
    """
    Wrapper for multiprocessing.
    """
    return process_centerline_line(line, angle_threshold, buffer_distance)

def process_centerlines(centerline_gdf: gpd.GeoDataFrame, angle_threshold: float = 130,
                        buffer_distance: float = 2, num_workers: int = 4) -> gpd.GeoDataFrame:
    """
    Process and smooth all centerlines using multiprocessing.
    """
    worker_function = partial(process_centerlines_worker, angle_threshold=angle_threshold, buffer_distance=buffer_distance)
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(worker_function, centerline_gdf["geometry"]),
                            total=len(centerline_gdf),
                            desc="Smoothing centerlines"))
    centerline_gdf = centerline_gdf.copy()
    centerline_gdf["geometry"] = results
    return centerline_gdf

# ----------------------------
# Flexible file reading helpers
# ----------------------------
def read_vector_file(path: str, layer: str = None) -> gpd.GeoDataFrame:
    """
    Reads a vector file. If a layer is provided (for a GeoPackage with multiple layers),
    attempts to read that layer; if it fails, logs a warning and reads without a layer.
    """
    if path.startswith("file://"):
        path = path[7:]
    if layer:
        try:
            return gpd.read_file(path, layer=layer)
        except Exception as e:
            logging.warning(f"Could not read layer '{layer}' from {path}: {e}. Trying without layer.")
            return gpd.read_file(path)
    return gpd.read_file(path)

def update_path_with_suffix(input_path: str, suffix: str) -> str:
    """
    Update the input path to include a suffix before the extension and
    return the new file path in the same folder as the original.
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
# Main processing function
# ---------------------------
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.logging.level)
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Extract assign_ID and smoothening parameters from configuration
    assign_cfg = cfg.assign_ID
    smooth_cfg = cfg.smoothening

    # Build centerline input from assign_ID output if requested.
    if smooth_cfg.get("use_assign_id_output", False):
        # Build path as in assign_ID: add '_centerline_ID' suffix.
        centerline_path = update_path_with_suffix(assign_cfg.centerline, "_centerline_ID")
        centerline_layer = assign_cfg.get("centerline_layer", None)
        output_dir = os.path.dirname(centerline_path)
        logging.info("Using assign_ID output for smoothing centerlines; output will be saved in the same folder as input.")
    else:
        centerline_path = smooth_cfg.centerline
        centerline_layer = smooth_cfg.get("centerline_layer", None)
        output_dir = smooth_cfg.output_dir

    angle_threshold = smooth_cfg.angle_threshold
    buffer_distance = smooth_cfg.buffer_distance
    num_workers = smooth_cfg.num_workers if smooth_cfg.num_workers > 0 else min(cpu_count(), 8)

    logging.info(f"Centerline Input Path: {centerline_path} | Layer: {centerline_layer}")
    logging.info(f"Output Directory: {output_dir}")
    logging.info(f"Processing parameters: angle_threshold={angle_threshold}, buffer_distance={buffer_distance}, num_workers={num_workers}")

    # Read the centerline vector file (if layer not found, will try without layer)
    centerline_gdf = read_vector_file(centerline_path, layer=centerline_layer)
    logging.info(f"Read {len(centerline_gdf)} centerline features.")

    # Process and smooth the centerlines using multiprocessing
    logging.info("Smoothing centerlines...")
    original_gdf = centerline_gdf.copy()
    smoothed_gdf = process_centerlines(centerline_gdf, angle_threshold=angle_threshold,
                                       buffer_distance=buffer_distance, num_workers=num_workers)

    # Remove any conflicting 'fid' fields
    for fid in ['fid', 'FID']:
        if fid in smoothed_gdf.columns:
            smoothed_gdf = smoothed_gdf.drop(columns=[fid])

    # Build output file path: add '_smooth' suffix to the centerline input file name, saved in the same folder.
    output_path = update_path_with_suffix(centerline_path, "_smooth")

    try:
        smoothed_gdf.to_file(output_path, driver="GPKG")
        logging.info(f"Smoothed centerlines saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
