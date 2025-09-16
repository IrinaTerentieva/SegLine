import os
import math
import logging
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, GeometryCollection, Point
from shapely.ops import split, linemerge, substring, unary_union
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import warnings

warnings.filterwarnings('ignore')


# -----------------------------
# Geometry helper functions
# -----------------------------
def extend_line(line: LineString, extension_distance=100):
    """
    Extend a line at both ends by a given distance.
    Handles MultiLineString by merging.
    """
    if isinstance(line, MultiLineString):
        line = linemerge(line)
    if not isinstance(line, LineString) or line.is_empty:
        return line
    try:
        coords = list(line.coords)
        if len(coords) < 2:
            return line

        # Extend at the start
        start_x, start_y = coords[0]
        next_x, next_y = coords[1]
        dx, dy = start_x - next_x, start_y - next_y
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            return line
        start_extension = (
            start_x + (dx / length) * extension_distance,
            start_y + (dy / length) * extension_distance,
        )

        # Extend at the end
        end_x, end_y = coords[-1]
        prev_x, prev_y = coords[-2]
        dx, dy = end_x - prev_x, end_y - prev_y
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            return line
        end_extension = (
            end_x + (dx / length) * extension_distance,
            end_y + (dy / length) * extension_distance,
        )

        extended_coords = [start_extension] + coords + [end_extension]
        return LineString(extended_coords)
    except Exception as e:
        logging.debug(f"Error extending line: {e}")
        return line


def generate_perpendiculars(centerline, avg_width, splitting_method, target_area, target_length, max_splitter_length=10):
    """
    Generate perpendicular lines to the centerline at intervals calculated to achieve a target area.
    """
    if avg_width <= 0:
        avg_width = 5

    # Set spacing, either by width or by area
    if splitting_method == "length":
        spacing = target_length
    elif splitting_method == "area":
        spacing = target_area / avg_width

    if spacing <= 0 or centerline.length <= 0:
        return []

    perpendiculars = []
    try:
        for distance in np.arange(0, centerline.length, spacing):
            point = centerline.interpolate(distance)
            next_point = centerline.interpolate(min(distance + 1, centerline.length))
            dx, dy = next_point.x - point.x, next_point.y - point.y
            perpendicular_vector = (-dy, dx)
            length = np.sqrt(perpendicular_vector[0] ** 2 + perpendicular_vector[1] ** 2)
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
        logging.debug(f"Error in generate_perpendiculars: {e}")
    return perpendiculars


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
        logging.debug(f"Error splitting geometry: {e}")
        return []


def process_polygon_worker(args):
    """
    Worker function for multiprocessing. Unpacks arguments and processes a single polygon.
    """
    (idx, footprint_row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id, splitting_method,
     target_area, target_length, extension_distance, crs) = args

    try:
        # Reconstruct geometry from WKT
        from shapely import wkt
        polygon = wkt.loads(geometry_wkt)

        unique_id = footprint_row_dict["UniqueID"]
        avg_width = footprint_row_dict.get("avg_width", 0)
        max_width = avg_width + 10

        if max_width <= 5:
            max_width = 15
        if avg_width >= 9:
            target_area = int(target_area * 2)  # For wide lines, increase plot area (area-based splitting only)

        # Get centerlines from dictionaries
        centerline_wkt = centerlines_by_id.get(unique_id)
        smooth_centerline_wkt = smooth_centerlines_by_id.get(unique_id)

        if not centerline_wkt or not smooth_centerline_wkt:
            return []

        # Reconstruct geometries
        centerline_geom = wkt.loads(centerline_wkt)
        smooth_centerline_geom = wkt.loads(smooth_centerline_wkt)

        if isinstance(smooth_centerline_geom, MultiLineString):
            smooth_centerline_geom = linemerge(smooth_centerline_geom)
        if isinstance(centerline_geom, MultiLineString):
            centerline_geom = linemerge(centerline_geom)

        extended_centerline = extend_line(centerline_geom, extension_distance)
        extended_smooth_centerline = extend_line(smooth_centerline_geom, extension_distance)

        # Try smooth centerline first
        perpendiculars = generate_perpendiculars(extended_smooth_centerline, avg_width, splitting_method,
                                                 target_area, target_length, max_splitter_length=max_width)

        # Fall back to regular centerline if needed
        if len(perpendiculars) < 5:
            perpendiculars = generate_perpendiculars(extended_centerline, avg_width, splitting_method,
                                                     target_area, target_length, max_splitter_length=max_width)

        # Split polygon
        segments = split_geometry(polygon, extended_centerline)
        for perp in perpendiculars:
            temp_segments = []
            for segment in segments:
                temp_segments.extend(split_geometry(segment, perp))
            segments = temp_segments

        # Return results
        results = []
        for part_id, segment in enumerate(segments):
            result = footprint_row_dict.copy()
            result['geometry'] = segment.wkt  # Store as WKT for serialization
            result['PartID'] = part_id
            result['original_idx'] = idx
            results.append(result)

        return results

    except Exception as e:
        logging.error(f"Error processing polygon {idx}: {e}")
        return []


def process_polygons_parallel_optimized(footprint_gdf, centerline_gdf, smooth_centerline_gdf, splitting_method,
                                        target_area, target_length, output_path, extension_distance=50, max_workers=None):
    """
    Optimized parallel processing using multiprocessing.Pool instead of ProcessPoolExecutor.
    """
    if max_workers is None:
        max_workers = min(cpu_count(), 8)

    logging.info(f"Using {max_workers} workers for parallel processing")

    # Pre-process centerlines into dictionaries for faster lookup
    centerlines_by_id = {}
    smooth_centerlines_by_id = {}

    for _, row in centerline_gdf.iterrows():
        if row['UniqueID'] and row.geometry:
            centerlines_by_id[row['UniqueID']] = row.geometry.wkt

    for _, row in smooth_centerline_gdf.iterrows():
        if row['UniqueID'] and row.geometry:
            smooth_centerlines_by_id[row['UniqueID']] = row.geometry.wkt

    # Prepare arguments for workers
    worker_args = []
    for idx, row in footprint_gdf.iterrows():
        if row.geometry and row.geometry.is_valid:
            # Convert row to dict without geometry
            row_dict = row.drop('geometry').to_dict()
            # Store geometry as WKT for serialization
            geometry_wkt = row.geometry.wkt

            args = (idx, row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id, splitting_method,
                    target_area, target_length, extension_distance, footprint_gdf.crs)
            worker_args.append(args)

    # Process in parallel
    results = []
    with Pool(processes=max_workers) as pool:
        # Use imap for better memory efficiency and progress tracking
        for result_batch in tqdm(pool.imap(process_polygon_worker, worker_args, chunksize=10),
                                 total=len(worker_args), desc="Processing polygons"):
            if result_batch:
                results.extend(result_batch)

    # Convert results back to GeoDataFrame
    if results:
        from shapely import wkt

        # Convert WKT geometries back to shapely objects
        for result in results:
            result['geometry'] = wkt.loads(result['geometry'])

        split_polygons_gdf = gpd.GeoDataFrame(results, crs=footprint_gdf.crs)

        # Calculate areas
        split_polygons_gdf['area'] = split_polygons_gdf.geometry.area

        # Drop temporary columns
        if 'original_idx' in split_polygons_gdf.columns:
            split_polygons_gdf = split_polygons_gdf.drop(columns=['original_idx'])

        # Save results
        split_polygons_gdf.to_file(output_path, driver="GPKG")
        logging.info(f"Split polygons saved to: {output_path}")
        logging.info(f"Created {len(split_polygons_gdf)} segments from {len(footprint_gdf)} polygons")
    else:
        logging.warning("No results generated from polygon splitting")


# ----------------------------
# File reading helpers
# ----------------------------
def read_vector_file(path: str, layer: str = None) -> gpd.GeoDataFrame:
    """
    Reads a vector file with optional layer specification.
    """
    if path.startswith("file://"):
        path = path[7:]
    try:
        if layer:
            return gpd.read_file(path, layer=layer)
        else:
            return gpd.read_file(path)
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        raise


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
# Main processing function
# ---------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up logging
    log_level = cfg.get("logging", {"level": "INFO"}).get("level", "INFO")
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Determine paths
    footprint_path = update_path_with_suffix(cfg.dataset.ground_footprint, "_footprint_ID")
    regular_centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID")

    # Get the smooth centerline path if smoothing is enabled
    if cfg.split_to_plots.get("use_smooth_centerline", True) and cfg.smoothening.get("perform_smoothing", True):
        smooth_centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID_smooth")
    else:
        smooth_centerline_path = regular_centerline_path

    # Configuration parameters
    splitting_method = cfg.params.splitting_method
    num_workers = cfg.split_to_plots.get("num_workers", None)
    segment_area = int(cfg.split_to_plots.segment_area)
    segment_length = int(cfg.split_to_plots.segment_length)
    extension_distance = cfg.split_to_plots.extension_distance
    max_splitter_length = cfg.split_to_plots.max_splitter_length_buffer


    # Output path
    output_dir = os.path.dirname(footprint_path)
    if splitting_method == "length":
        output_filename = os.path.basename(footprint_path).replace(".gpkg", f"_segments{segment_length}m.gpkg")
    elif splitting_method == "area":
        output_filename = os.path.basename(footprint_path).replace(".gpkg", f"_segments{segment_area}m2.gpkg")

    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"Footprint splitting method: {splitting_method}")
    logging.info(f"Footprint path: {footprint_path}")
    logging.info(f"Regular centerline path: {regular_centerline_path}")
    logging.info(f"Smooth centerline path: {smooth_centerline_path}")
    logging.info(f"Output path: {output_path}")
    if splitting_method == "length":
        logging.info(f"Parameters: segment_length={segment_length}m, extension={extension_distance}m")
    elif splitting_method == "area":
        logging.info(f"Parameters: segment_area={segment_area}m², extension={extension_distance}m")


    # Read input data
    try:
        logging.info("Reading footprint data...")
        footprint_gdf = read_vector_file(footprint_path)

        logging.info("Reading centerline data...")
        regular_centerline_gdf = read_vector_file(regular_centerline_path)
        smooth_centerline_gdf = read_vector_file(smooth_centerline_path)

        logging.info(f"Loaded {len(footprint_gdf)} footprints, {len(regular_centerline_gdf)} centerlines")

    except Exception as e:
        logging.error(f"Failed to read input files: {e}")
        return

    # Process polygons
    process_polygons_parallel_optimized(
        footprint_gdf,
        regular_centerline_gdf,
        smooth_centerline_gdf,
        splitting_method,
        segment_area,
        segment_length,
        output_path,
        extension_distance=extension_distance,
        max_workers=num_workers
    )

    logging.info("Processing complete!")


if __name__ == "__main__":
    main()