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


def generate_perpendiculars(centerline, spacing, max_splitter_length=10):
    """
    Generate perpendicular lines to the centerline at fixed intervals.

    Args:
        centerline: LineString geometry
        spacing: Fixed distance in meters between perpendiculars
        max_splitter_length: Maximum length of perpendicular lines
    """
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


def process_subplot_worker(args):
    """
    Worker function for multiprocessing.
    Processes one plot polygon and splits it into subplots.
    plot_id will be reassigned to globally unique integer after all processing.
    """
    (idx, footprint_row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id,
     spacing, extension_distance, width_column, crs) = args

    try:
        # Reconstruct geometry from WKT
        from shapely import wkt
        polygon = wkt.loads(geometry_wkt)

        unique_id = footprint_row_dict["UniqueID"]

        # Keep the original plot_id for temporary tracking
        # This will be replaced with globally unique integer after all processing
        original_plot_id = footprint_row_dict.get("plot_id", idx)

        # Get width column name from config (passed as parameter)
        avg_width = footprint_row_dict.get(width_column, 0)
        if avg_width is None:
            avg_width = 0

        max_width = avg_width + 10

        if max_width <= 5:
            max_width = 15

        # Auto-estimate width from geometry if needed
        centerline_wkt_temp = centerlines_by_id.get(unique_id)
        if centerline_wkt_temp:
            centerline_temp = wkt.loads(centerline_wkt_temp)
            if centerline_temp.length > 0:
                estimated_width = polygon.area / centerline_temp.length
                if estimated_width > max_width:
                    max_width = estimated_width + 10  # Add buffer

        # Get centerlines from dictionaries
        centerline_wkt = centerlines_by_id.get(unique_id)
        smooth_centerline_wkt = smooth_centerlines_by_id.get(unique_id)

        if not centerline_wkt or not smooth_centerline_wkt:
            logging.debug(f"No centerlines for UniqueID: {unique_id}")
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
        perpendiculars = generate_perpendiculars(extended_smooth_centerline, spacing,
                                                 max_splitter_length=max_width)

        # Fall back to regular centerline if needed
        if len(perpendiculars) < 5:
            logging.debug(f'Bad perpendiculars for {unique_id}, switching to regular centerline')
            perpendiculars = generate_perpendiculars(extended_centerline, spacing,
                                                     max_splitter_length=max_width)

        # Split polygon into subplots
        segments = split_geometry(polygon, extended_centerline)
        if not segments:
            # If centerline split fails, use original polygon
            segments = [polygon]

        for perp in perpendiculars:
            temp_segments = []
            for segment in segments:
                split_result = split_geometry(segment, perp)
                if split_result:
                    temp_segments.extend(split_result)
                else:
                    # If split fails, keep the original segment
                    temp_segments.append(segment)
            segments = temp_segments

        # Create results
        # Store original grouping info - plot_id will be reassigned to be globally unique
        results = []
        for subplot_idx, segment in enumerate(segments):
            if segment.area > 0:  # Only include valid segments
                result = footprint_row_dict.copy()
                result['geometry'] = segment.wkt  # Store as WKT for serialization

                # Keep temporary grouping identifier for processing
                # Format: "UniqueID_original_plot_id" for sorting/grouping
                result['temp_group_id'] = f"{unique_id}_{original_plot_id}"
                result['original_plot_id'] = original_plot_id

                # Subplot number within this plot
                result['SubplotPartID'] = subplot_idx

                result['original_idx'] = idx
                results.append(result)

        return results

    except Exception as e:
        logging.error(f"Error processing subplot {idx}: {e}")
        return []


def process_subplots_parallel_optimized(footprint_gdf, centerline_gdf, smooth_centerline_gdf,
                                        spacing, output_path, extension_distance, width_column, max_workers=None):
    """
    Optimized parallel processing using multiprocessing.Pool.
    ✅ Assigns globally unique INTEGER plot_id and subplot_id after processing.
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

    # Log statistics
    logging.info(f"Loaded {len(centerlines_by_id)} regular centerlines")
    logging.info(f"Loaded {len(smooth_centerlines_by_id)} smooth centerlines")

    # Check input
    if 'plot_id' in footprint_gdf.columns:
        n_plots = len(footprint_gdf)
        n_unique_plot_ids = footprint_gdf['plot_id'].nunique()
        n_unique_ids = footprint_gdf['UniqueID'].nunique()

        logging.info(f"Input has {n_plots} plots from {n_unique_ids} UniqueIDs")
        logging.info(
            f"  - plot_id values range from {footprint_gdf['plot_id'].min()} to {footprint_gdf['plot_id'].max()}")
        logging.info(f"  - {n_unique_plot_ids} unique plot_id values (may have duplicates across UniqueIDs)")

        # Check for duplicates
        if n_plots != n_unique_plot_ids:
            logging.info(f"  → plot_id is NOT globally unique - will create globally unique integers")
        else:
            logging.info(f"  → plot_id appears to be globally unique already")

    # Prepare arguments for workers
    worker_args = []
    for idx, row in footprint_gdf.iterrows():
        if row.geometry and row.geometry.is_valid:
            # Convert row to dict without geometry
            row_dict = row.drop('geometry').to_dict()
            # Store geometry as WKT for serialization
            geometry_wkt = row.geometry.wkt

            args = (idx, row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id,
                    spacing, extension_distance, width_column, footprint_gdf.crs)
            worker_args.append(args)

    logging.info(f"Processing {len(worker_args)} plots into subplots...")

    # Process in parallel
    results = []
    successful_count = 0

    with Pool(processes=max_workers) as pool:
        # Use imap for better memory efficiency and progress tracking
        for result_batch in tqdm(pool.imap(process_subplot_worker, worker_args, chunksize=10),
                                 total=len(worker_args), desc="Processing subplots"):
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

        split_polygons_gdf = gpd.GeoDataFrame(results, crs=footprint_gdf.crs)

        # ✅ ASSIGN GLOBALLY UNIQUE INTEGER plot_id
        # Sort by UniqueID and temp_group_id to maintain consistent ordering
        split_polygons_gdf = split_polygons_gdf.sort_values(['UniqueID', 'temp_group_id', 'SubplotPartID']).reset_index(
            drop=True)

        # Create mapping from temp_group_id to globally unique plot_id
        unique_groups = split_polygons_gdf['temp_group_id'].unique()
        group_to_plot_id = {group: i for i, group in enumerate(unique_groups)}

        # Assign globally unique plot_id (integer)
        split_polygons_gdf['plot_id'] = split_polygons_gdf['temp_group_id'].map(group_to_plot_id)

        # ✅ ASSIGN GLOBALLY UNIQUE INTEGER subplot_id
        # Simply use the index after sorting
        split_polygons_gdf['subplot_id'] = split_polygons_gdf.index

        # Calculate areas and lengths
        split_polygons_gdf['area'] = split_polygons_gdf.geometry.area
        split_polygons_gdf['length'] = split_polygons_gdf.geometry.length
        split_polygons_gdf['perimeter'] = split_polygons_gdf.geometry.length

        # Drop temporary columns
        temp_cols = ['original_idx', 'temp_group_id', 'original_plot_id']
        split_polygons_gdf = split_polygons_gdf.drop(
            columns=[col for col in temp_cols if col in split_polygons_gdf.columns])

        # ✅ VERIFICATION
        n_subplots = len(split_polygons_gdf)
        n_unique_plot_ids = split_polygons_gdf['plot_id'].nunique()
        n_unique_subplot_ids = split_polygons_gdf['subplot_id'].nunique()
        n_unique_ids = split_polygons_gdf['UniqueID'].nunique()

        logging.info("")
        logging.info("=" * 70)
        logging.info("VERIFICATION RESULTS:")
        logging.info("=" * 70)
        logging.info(f"Total subplots created: {n_subplots}")
        logging.info(f"From {n_unique_ids} UniqueIDs")
        logging.info(
            f"Across {n_unique_plot_ids} unique plot_ids (integers: {split_polygons_gdf['plot_id'].min()} to {split_polygons_gdf['plot_id'].max()})")
        logging.info(
            f"With {n_unique_subplot_ids} unique subplot_ids (integers: {split_polygons_gdf['subplot_id'].min()} to {split_polygons_gdf['subplot_id'].max()})")
        logging.info("")

        # Verify uniqueness
        if n_unique_plot_ids == len(unique_groups):
            logging.info(f"✓ plot_id is GLOBALLY UNIQUE (no duplicates across all UniqueIDs)")
        else:
            logging.warning(f"⚠ plot_id issue detected")

        if n_subplots == n_unique_subplot_ids:
            logging.info(f"✓ subplot_id is GLOBALLY UNIQUE (no duplicates)")
        else:
            logging.warning(f"⚠ subplot_id has duplicates")

        # Show sample data
        # XkXepP
        # split_polygons_gdf = split_polygons_gdf[split_polygons_gdf['UniqueID'] == 'XkXepP']
        sample_data = split_polygons_gdf[['UniqueID', 'plot_id', 'subplot_id', 'SubplotPartID']].head(10)
        logging.info("")
        logging.info("Sample data (first 10 rows):")
        logging.info(sample_data.to_string())

        # Distribution statistics
        plots_per_unique_id = split_polygons_gdf.groupby('UniqueID')['plot_id'].nunique()
        subplots_per_plot = split_polygons_gdf.groupby('plot_id').size()

        logging.info("")
        logging.info(f"Plots per UniqueID - Min: {plots_per_unique_id.min()}, "
                     f"Mean: {plots_per_unique_id.mean():.1f}, "
                     f"Median: {plots_per_unique_id.median():.0f}, "
                     f"Max: {plots_per_unique_id.max()}")
        logging.info(f"Subplots per plot - Min: {subplots_per_plot.min()}, "
                     f"Mean: {subplots_per_plot.mean():.1f}, "
                     f"Median: {subplots_per_plot.median():.0f}, "
                     f"Max: {subplots_per_plot.max()}")

        # Show example of one UniqueID
        example_uid = split_polygons_gdf['UniqueID'].iloc[0]
        example_data = split_polygons_gdf[split_polygons_gdf['UniqueID'] == example_uid]
        example_plot_ids = sorted(example_data['plot_id'].unique())
        logging.info("")
        logging.info(
            f"Example: UniqueID '{example_uid}' has {len(example_plot_ids)} plots with plot_ids: {example_plot_ids[:10]}{' ...' if len(example_plot_ids) > 10 else ''}")

        logging.info("=" * 70)
        logging.info("")

        # Save results
        split_polygons_gdf.to_file(output_path, driver="GPKG")
        logging.info(f"Split polygons saved to: {output_path}")

        # Log statistics
        avg_area = split_polygons_gdf['area'].mean()
        median_area = split_polygons_gdf['area'].median()
        avg_length = split_polygons_gdf['length'].mean()
        median_length = split_polygons_gdf['length'].median()
        logging.info(f"Average subplot area: {avg_area:.2f} m²")
        logging.info(f"Median subplot area: {median_area:.2f} m²")
        logging.info(f"Average perimeter: {avg_length:.2f} m")
        logging.info(f"Median perimeter: {median_length:.2f} m")

    else:
        logging.warning("No results generated from subplot splitting")


def plot_perpendiculars(centerline, perpendiculars, output_path):
    """
    Plot centerline and perpendiculars for debugging.
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
    logging.info(f"Plot saved to {output_path}")


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

    # Get spacing from split_to_plots (used in filename)
    plot_spacing = int(cfg.split_to_plots.perpendicular_spacing)

    # Build input path from split_to_sides output
    # This should be: {base}_footprint_ID_segments{plot_spacing}m_sides.gpkg
    footprint_base = update_path_with_suffix(cfg.dataset.ground_footprint, "_footprint_ID")
    footprint_path = footprint_base.replace(".gpkg", f"_segments{plot_spacing}m_sides.gpkg")

    # Get centerline paths based on configuration
    if cfg.split_to_subplots.get("use_smooth_centerline", True) and cfg.smoothening.get("perform_smoothing", True):
        centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID")
        smooth_centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID_smooth")
    else:
        # If no smoothing, use regular centerline for both
        centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID")
        smooth_centerline_path = centerline_path

    # Configuration parameters
    num_workers = cfg.split_to_subplots.get("num_workers", None)
    subplot_spacing = float(cfg.split_to_subplots.perpendicular_spacing)
    extension_distance = cfg.split_to_plots.extension_distance
    print('Extension distance: ', extension_distance)

    max_splitter_length = cfg.split_to_subplots.max_splitter_length_buffer

    # Output path - build from the _sides.gpkg input path
    output_dir = os.path.dirname(footprint_path)
    output_filename = os.path.basename(footprint_path).replace("_sides.gpkg", f"_subplots{int(subplot_spacing)}m.gpkg")
    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"Input footprint (sides) path: {footprint_path}")
    logging.info(f"Regular centerline path: {centerline_path}")
    logging.info(f"Smooth centerline path: {smooth_centerline_path}")
    logging.info(f"Output path: {output_path}")
    logging.info(f"Parameters: perpendicular_spacing={subplot_spacing}m, extension={extension_distance}m")

    # Check if input file exists
    if not os.path.exists(footprint_path):
        logging.error(f"Input file does not exist: {footprint_path}")
        logging.error("Make sure split_to_sides has been run first!")
        return

    # Read input data
    try:
        logging.info("Reading footprint data...")
        footprint_gdf = read_vector_file(footprint_path)

        logging.info("Reading centerline data...")
        centerline_gdf = read_vector_file(centerline_path)
        smooth_centerline_gdf = read_vector_file(smooth_centerline_path)

        logging.info(f"Loaded {len(footprint_gdf)} footprints, {len(centerline_gdf)} centerlines")

    except Exception as e:
        logging.error(f"Failed to read input files: {e}")
        return

    # Process subplots
    process_subplots_parallel_optimized(
        footprint_gdf,
        centerline_gdf,
        smooth_centerline_gdf,
        subplot_spacing,
        output_path,
        extension_distance=extension_distance,
        width_column=cfg.dataset.width_column,
        max_workers=num_workers
    )

    logging.info("Subplot processing complete!")


if __name__ == "__main__":
    main()