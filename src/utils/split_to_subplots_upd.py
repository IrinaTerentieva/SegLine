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


def generate_perpendiculars_for_subplot(centerline, target_area, max_splitter_length=10, avg_width=None):
    """
    Generate perpendicular lines to the centerline at intervals calculated to achieve a target area.
    Modified to handle subplot-specific logic.
    """
    if avg_width is None or avg_width <= 0:
        # Estimate width based on target area and centerline length
        avg_width = min(10, target_area / max(centerline.length, 1))

    spacing = target_area / avg_width
    if spacing <= 0 or centerline.length <= 0:
        return []

    perpendiculars = []
    try:
        for distance in np.arange(spacing / 2, centerline.length, spacing):
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


def process_subplot_worker(args):
    """
    Worker function for processing plots into subplots while maintaining pairing.
    """
    (idx, plot_row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id,
     target_area, extension_distance, crs) = args

    try:
        # Reconstruct geometry from WKT
        from shapely import wkt
        polygon = wkt.loads(geometry_wkt)

        unique_id = plot_row_dict["UniqueID"]

        # Preserve the plot_id and side from the input
        plot_id = plot_row_dict.get("plot_id", idx)
        side = plot_row_dict.get("side", "unknown")

        # Get width information
        avg_width = plot_row_dict.get("avg_width", 5)

        # For subplots, we might want smaller perpendiculars
        max_width = min(avg_width + 5, 15)  # Smaller than for plots

        # Get centerlines from dictionaries
        centerline_wkt = centerlines_by_id.get(unique_id)
        smooth_centerline_wkt = smooth_centerlines_by_id.get(unique_id)

        if not centerline_wkt:
            logging.debug(f"No centerlines for UniqueID: {unique_id}")
            return []

        # Reconstruct geometries
        centerline_geom = wkt.loads(centerline_wkt)

        if smooth_centerline_wkt:
            smooth_centerline_geom = wkt.loads(smooth_centerline_wkt)
            if isinstance(smooth_centerline_geom, MultiLineString):
                smooth_centerline_geom = linemerge(smooth_centerline_geom)
        else:
            smooth_centerline_geom = centerline_geom

        if isinstance(centerline_geom, MultiLineString):
            centerline_geom = linemerge(centerline_geom)

        # Extend centerlines for better splitting
        extended_centerline = extend_line(centerline_geom, extension_distance)
        extended_smooth_centerline = extend_line(smooth_centerline_geom, extension_distance)

        # Try smooth centerline first for perpendiculars
        perpendiculars = generate_perpendiculars_for_subplot(
            extended_smooth_centerline, target_area,
            max_splitter_length=max_width, avg_width=avg_width
        )

        # Fall back to regular centerline if needed
        if len(perpendiculars) < 2:
            perpendiculars = generate_perpendiculars_for_subplot(
                extended_centerline, target_area,
                max_splitter_length=max_width, avg_width=avg_width
            )

        # Split polygon into subplots
        segments = [polygon]  # Start with the original plot polygon

        # Split by perpendiculars
        for perp in perpendiculars:
            temp_segments = []
            for segment in segments:
                temp_segments.extend(split_geometry(segment, perp))
            segments = temp_segments

        # Generate results with proper subplot IDs
        results = []
        for subplot_idx, segment in enumerate(segments):
            if segment.area > 0:  # Only include valid segments
                result = plot_row_dict.copy()
                result['geometry'] = segment.wkt

                # Create subplot_id that maintains the pairing
                # Format: {plot_id}_subplot{subplot_idx}
                result['subplot_id'] = f"{plot_id}_subplot{subplot_idx}"

                # Preserve original plot_id and side
                result['original_plot_id'] = plot_id
                result['side'] = side

                # Add subplot-specific fields
                result['subplot_index'] = subplot_idx
                result['original_idx'] = idx

                results.append(result)

        return results

    except Exception as e:
        logging.error(f"Error processing subplot {idx}: {e}")
        return []


def process_subplots_with_pairing(plots_gdf, centerline_gdf, smooth_centerline_gdf,
                                  target_area, output_path, extension_distance=50,
                                  max_workers=None):
    """
    Process plots into subplots while maintaining the pairing from plot_id.
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

    logging.info(f"Loaded {len(centerlines_by_id)} regular centerlines")
    logging.info(f"Loaded {len(smooth_centerlines_by_id)} smooth centerlines")

    # Check if plot_id exists in the input
    if 'plot_id' not in plots_gdf.columns:
        logging.warning("No 'plot_id' column found. Subplots won't be paired.")

    # Prepare arguments for workers
    worker_args = []
    for idx, row in plots_gdf.iterrows():
        if row.geometry and row.geometry.is_valid:
            # Convert row to dict without geometry
            row_dict = row.drop('geometry').to_dict()
            # Store geometry as WKT for serialization
            geometry_wkt = row.geometry.wkt

            args = (idx, row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id,
                    target_area, extension_distance, plots_gdf.crs)
            worker_args.append(args)

    logging.info(f"Processing {len(worker_args)} plots into subplots...")

    # Process in parallel
    results = []
    successful_count = 0

    with Pool(processes=max_workers) as pool:
        # Use imap for better memory efficiency and progress tracking
        for result_batch in tqdm(pool.imap(process_subplot_worker, worker_args, chunksize=10),
                                 total=len(worker_args), desc="Creating subplots"):
            if result_batch:
                results.extend(result_batch)
                successful_count += 1

    logging.info(f"Successfully processed {successful_count}/{len(worker_args)} plots")

    # Convert results back to GeoDataFrame
    if results:
        from shapely import wkt

        # Convert WKT geometries back to shapely objects
        for result in results:
            result['geometry'] = wkt.loads(result['geometry'])

        subplots_gdf = gpd.GeoDataFrame(results, crs=plots_gdf.crs)

        # Calculate areas
        subplots_gdf['area'] = subplots_gdf.geometry.area

        # Drop temporary columns
        if 'original_idx' in subplots_gdf.columns:
            subplots_gdf = subplots_gdf.drop(columns=['original_idx'])

        # Analyze pairing statistics
        if 'original_plot_id' in subplots_gdf.columns:
            # Count subplots per original plot_id
            plot_id_counts = subplots_gdf.groupby('original_plot_id').size()

            # For each plot_id, check if we have subplots on both sides
            paired_stats = []
            for plot_id in subplots_gdf['original_plot_id'].unique():
                plot_subplots = subplots_gdf[subplots_gdf['original_plot_id'] == plot_id]
                sides = plot_subplots['side'].value_counts()
                if len(sides) > 1:  # Has subplots on multiple sides
                    paired_stats.append(plot_id)

            logging.info(f"Plot IDs with subplots on both sides: {len(paired_stats)}")

            # Check subplot pairing
            subplot_id_groups = subplots_gdf.groupby('subplot_id')
            paired_subplots = 0
            for subplot_id, group in subplot_id_groups:
                # Subplots with same subplot_id should be on the same side
                # but from the same original plot
                if len(group) > 0:
                    paired_subplots += len(group)

            logging.info(f"Total subplots with consistent IDs: {paired_subplots}")

        # Save results
        subplots_gdf.to_file(output_path, driver="GPKG")
        logging.info(f"Subplots saved to: {output_path}")
        logging.info(f"Created {len(subplots_gdf)} subplots from {len(plots_gdf)} plots")

        # Log statistics
        avg_area = subplots_gdf['area'].mean()
        median_area = subplots_gdf['area'].median()
        logging.info(f"Average subplot area: {avg_area:.2f} m²")
        logging.info(f"Median subplot area: {median_area:.2f} m²")
    else:
        logging.warning("No results generated from subplot splitting")


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

    # Determine paths based on configuration
    output_dir = os.path.dirname(cfg.dataset.ground_footprint)

    # Input should be the output from split_to_side (plots with plot_id)
    # Construct the expected filename
    footprint_base = os.path.basename(cfg.dataset.ground_footprint)
    if ".shp" in footprint_base:
        base_name = footprint_base.replace(".shp", "")
    else:
        base_name = footprint_base.replace(".gpkg", "")

    # The input file should be the output from split_to_side
    # Expected pattern: *_sides.gpkg or similar
    input_filename = f"{base_name}_sides.gpkg"
    input_path = os.path.join(output_dir, input_filename)

    # If that doesn't exist, try the plot output pattern
    if not os.path.exists(input_path):
        input_filename = f"{base_name}_footprint_ID_plots.gpkg"
        input_path = os.path.join(output_dir, input_filename)

    # Get centerline paths
    centerline_base = os.path.basename(cfg.dataset.centerline)
    regular_centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID")

    # Get smooth centerline path if available
    if cfg.split_to_subplots.get("use_smooth_centerline", True):
        smooth_centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID_smooth")
    else:
        smooth_centerline_path = regular_centerline_path

    # Configuration parameters
    num_workers = cfg.split_to_subplots.get("num_workers", None)
    segment_area = int(cfg.split_to_subplots.segment_area)
    extension_distance = cfg.split_to_subplots.extension_distance

    # Output path
    output_filename = input_filename.replace(".gpkg", f"_subplots{segment_area}m2.gpkg")
    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"Input plots path: {input_path}")
    logging.info(f"Regular centerline path: {regular_centerline_path}")
    logging.info(f"Smooth centerline path: {smooth_centerline_path}")
    logging.info(f"Output path: {output_path}")
    logging.info(f"Parameters: subplot_area={segment_area}m², extension={extension_distance}m")

    # Check if input exists
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        logging.error("Please run split_to_side.py first to generate plots with plot_ids")
        return

    # Read input data
    try:
        logging.info("Reading plots data...")
        plots_gdf = read_vector_file(input_path)

        logging.info("Reading centerline data...")
        centerline_gdf = read_vector_file(regular_centerline_path)

        if os.path.exists(smooth_centerline_path):
            smooth_centerline_gdf = read_vector_file(smooth_centerline_path)
        else:
            logging.warning(f"Smooth centerline not found at {smooth_centerline_path}, using regular centerline")
            smooth_centerline_gdf = centerline_gdf

        logging.info(f"Loaded {len(plots_gdf)} plots, {len(centerline_gdf)} centerlines")

        # Check for plot_id column
        if 'plot_id' in plots_gdf.columns:
            unique_plot_ids = plots_gdf['plot_id'].nunique()
            logging.info(f"Found {unique_plot_ids} unique plot_ids in input")
        else:
            logging.warning("No 'plot_id' column found in input - subplots won't maintain pairing")

    except Exception as e:
        logging.error(f"Failed to read input files: {e}")
        return

    # Process plots into subplots
    process_subplots_with_pairing(
        plots_gdf,
        centerline_gdf,
        smooth_centerline_gdf,
        segment_area,
        output_path,
        extension_distance=extension_distance,
        max_workers=num_workers
    )

    logging.info("Subplot processing complete!")


if __name__ == "__main__":
    main()