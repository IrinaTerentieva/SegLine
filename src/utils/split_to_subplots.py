import os
import math
import logging
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, GeometryCollection, Point
from shapely.ops import split, linemerge, substring, unary_union
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import random

# -----------------------------
# Geometry helper functions
# -----------------------------
def extend_line(line: LineString, extension_distance=100):
    """
    Extend a line at both ends by a given distance.
    Handles MultiLineString by merging.
    """
    if isinstance(line, MultiLineString):
        print("Merging MultiLineString into LineString.")
        line = linemerge(line)
    if not isinstance(line, LineString) or line.is_empty:
        print(f"Invalid or unsupported geometry type: {type(line)}")
        return line
    try:
        coords = list(line.coords)
        # Extend at the start
        start_x, start_y = coords[0]
        next_x, next_y = coords[1]
        dx, dy = start_x - next_x, start_y - next_y
        length = np.sqrt(dx**2 + dy**2)
        start_extension = (
            start_x + (dx / length) * extension_distance,
            start_y + (dy / length) * extension_distance,
        )
        # Extend at the end
        end_x, end_y = coords[-1]
        prev_x, prev_y = coords[-2]
        dx, dy = end_x - prev_x, end_y - prev_y
        length = np.sqrt(dx**2 + dy**2)
        end_extension = (
            end_x + (dx / length) * extension_distance,
            end_y + (dy / length) * extension_distance,
        )
        extended_coords = [start_extension] + coords + [end_extension]
        return LineString(extended_coords)
    except Exception as e:
        print(f"Error extending line: {e}")
        return line

def generate_perpendiculars(centerline, avg_width, target_area, max_splitter_length=10):
    """
    Generate perpendicular lines to the centerline at intervals calculated to achieve a target area.
    If avg_width is non-positive, a default value is used.
    """
    if avg_width <= 0:
        print("Warning: avg_width is non-positive; using default value 5.")
        avg_width = 5
    spacing = target_area / avg_width
    if spacing <= 0 or centerline.length <= 0:
        print("Invalid spacing or centerline length.")
        return []
    perpendiculars = []
    try:
        for distance in np.arange(0, centerline.length, spacing):
            point = centerline.interpolate(distance)
            next_point = centerline.interpolate(min(distance + 1, centerline.length))
            dx, dy = next_point.x - point.x, next_point.y - point.y
            perpendicular_vector = (-dy, dx)
            length = np.sqrt(perpendicular_vector[0]**2 + perpendicular_vector[1]**2)
            if length == 0:
                continue
            unit_vector = (perpendicular_vector[0] / length, perpendicular_vector[1] / length)
            half_length = max_splitter_length / 2
            start = (point.x - unit_vector[0] * half_length,
                     point.y - unit_vector[1] * half_length)
            end = (point.x + unit_vector[0] * half_length,
                   point.y + unit_vector[1] * half_length)
            perpendicular_line = LineString([start, end])
            perpendiculars.append(perpendicular_line)
    except Exception as e:
        print(f"Error in generate_perpendiculars: {e}")
    return perpendiculars

def plot_perpendiculars(centerline, perpendiculars, output_path):
    """
    Plot centerline and perpendiculars, and save the visualization to a PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(centerline, LineString):
        x, y = centerline.xy
        ax.plot(x, y, label="Centerline", color="blue", linewidth=2)
    for perp in perpendiculars:
        if isinstance(perp, LineString):
            px, py = perp.xy
            ax.plot(px, py, color="red", linewidth=1)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Centerline and Perpendiculars")
    ax.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")

def split_geometry(geometry, splitter):
    """
    Split a geometry with a splitter, handling GeometryCollection properly.
    """
    try:
        result = split(geometry, splitter)
        if isinstance(result, GeometryCollection):
            return [geom for geom in result.geoms if isinstance(geom, Polygon)]
        elif isinstance(result, Polygon):
            return [result]
        else:
            return []
    except Exception as e:
        print(f"Error splitting geometry: {e}")
        return []

def process_polygon(footprint_row, centerline_gdf, smooth_centerline_gdf, target_area, extension_distance=50):
    """
    Process a single polygon using centerline and its smoothed version.
    """
    unique_id = footprint_row["UniqueID"]
    polygon = footprint_row.geometry
    avg_width = footprint_row.get("avg_width", 0)
    max_width = avg_width + 10
    print(unique_id)
    if max_width <= 5:
        max_width = 15
    if avg_width >= 9:
        target_area = int(target_area * 2)
    matched_smooth_centerline = smooth_centerline_gdf[smooth_centerline_gdf["UniqueID"] == unique_id]
    matched_centerline = centerline_gdf[centerline_gdf["UniqueID"] == unique_id]
    if matched_smooth_centerline.empty or matched_centerline.empty:
        print(f"No centerlines for UniqueID: {unique_id}")
        return []
    smooth_centerline_geom = matched_smooth_centerline.iloc[0].geometry
    centerline_geom = matched_centerline.iloc[0].geometry
    if isinstance(smooth_centerline_geom, MultiLineString):
        smooth_centerline_geom = linemerge(smooth_centerline_geom)
    if isinstance(centerline_geom, MultiLineString):
        centerline_geom = linemerge(centerline_geom)
    extended_centerline = extend_line(centerline_geom, extension_distance)
    extended_smooth_centerline = extend_line(smooth_centerline_geom, extension_distance)
    try:
        perpendiculars = generate_perpendiculars(extended_smooth_centerline, avg_width, target_area, max_splitter_length=max_width)
    except Exception as e:
        perpendiculars = generate_perpendiculars(extended_centerline, avg_width, target_area, max_splitter_length=max_width)
    if len(perpendiculars) < 5:
        print('Bad perpendiculars, switch to regular centerline')
        perpendiculars = generate_perpendiculars(extended_centerline, avg_width, target_area, max_splitter_length=max_width)
    segments = split_geometry(polygon, extended_centerline)
    for perp in perpendiculars:
        temp_segments = []
        for segment in segments:
            temp_segments.extend(split_geometry(segment, perp))
        segments = temp_segments
    return [
        {**footprint_row.drop("geometry"), "geometry": segment, "PartID": part_id}
        for part_id, segment in enumerate(segments)
    ]

def process_polygons_parallel(footprint_gdf, centerline_gdf, smooth_centerline_gdf, target_area, output_path, max_workers=4):
    """
    Process polygons in parallel using multiprocessing.
    """
    results = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_polygon, row, centerline_gdf, smooth_centerline_gdf, target_area
            )
            for _, row in footprint_gdf.iterrows()
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing polygons"):
            result = future.result()
            if result:
                results.extend(result)
    split_polygons_gdf = gpd.GeoDataFrame(results, crs=footprint_gdf.crs)
    split_polygons_gdf['area'] = split_polygons_gdf['geometry'].area
    split_polygons_gdf = split_polygons_gdf.reset_index(drop=True)
    split_polygons_gdf['plot_id'] = split_polygons_gdf.index
    split_polygons_gdf.to_file(output_path, driver="GPKG")
    print(f"Split polygons saved to: {output_path}")

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
# Main processing function using Hydra
# ---------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up logging; use default INFO level if not provided in config.
    log_level = cfg.get("logging", {"level": "INFO"}).get("level", "INFO")
    logging.basicConfig(level=log_level)
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Determine paths:
    # Footprint comes from the UniqueID process.
    footprint_path = update_path_with_suffix(cfg.dataset.ground_footprint, "_sides")
    # For centerline, if splitting should use smoothed centerline (default true), then use the smoothed output;
    # otherwise, use the regular UniqueID centerline output.
    if cfg.split_to_subplots.get("use_smooth_centerline", True):
        centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID_smooth")
    else:
        centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID")
    # For splitting, we need both the real centerline and the smooth one.
    # If smoothing is not used, they are the same.
    if cfg.split_to_subplots.get("use_smooth_centerline", True):
        smooth_centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID_smooth")
    else:
        smooth_centerline_path = centerline_path

    # Use the same folder as the input file for output.
    output_dir = os.path.dirname(footprint_path)

    num_workers = cfg.split_to_subplots.num_workers
    segment_area = int(cfg.split_to_subplots.segment_area)
    extension_distance = cfg.split_to_subplots.extension_distance
    max_splitter_length = cfg.split_to_subplots.max_splitter_length_buffer

    output_filename = os.path.basename(footprint_path).replace("sides.gpkg", f"_subplots{segment_area}m2.gpkg")
    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"Footprint path (UniqueID output): {footprint_path}")
    logging.info(f"Centerline path: {centerline_path}")
    logging.info(f"Smooth centerline path: {smooth_centerline_path}")
    logging.info(f"Output path for split polygons: {output_path}")
    logging.info(f"Segment area: {segment_area} m2, Extension distance: {extension_distance}, Max splitter length: {max_splitter_length}")

    # Read input data
    footprint_gdf = read_vector_file(footprint_path, layer=cfg.dataset.get("ground_footprint_layer"))
    centerline_gdf = read_vector_file(centerline_path, layer=cfg.dataset.get("centerline_layer"))
    smooth_centerline_gdf = read_vector_file(smooth_centerline_path, layer=cfg.dataset.get("centerline_layer"))

    print(f"Splitting with target area {segment_area} m2")
    print("Smooth centerline:", smooth_centerline_path)
    print("Real centerline:", centerline_path)

    # Process polygons in parallel
    process_polygons_parallel(footprint_gdf, centerline_gdf, smooth_centerline_gdf, segment_area, output_path, max_workers=num_workers)
    print("Processing complete.")

if __name__ == "__main__":
    main()
