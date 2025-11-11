import os
import random
import string
import logging
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.errors import GEOSException
from shapely.geometry import Point
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

warnings.filterwarnings('ignore')

# Set the seed for reproducibility
random.seed(101)


def generate_random_id(length=8):
    """Generate a random UniqueID consisting of letters and numbers."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def fix_invalid_geometry(geometry):
    """Attempt to fix invalid geometries using buffer(0) and handle errors gracefully."""
    try:
        if geometry is None or geometry.is_empty:
            return None
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        return geometry
    except GEOSException as e:
        logging.error(f"Geometry error during fix: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during fix: {e}")
        return None


def drop_fid_column(gdf):
    """Drop 'fid' and 'FID' columns from a GeoDataFrame if they exist."""
    for fid in ['fid', 'FID']:
        if fid in gdf.columns:
            logging.info(f"Dropping '{fid}' column to avoid schema conflict.")
            gdf = gdf.drop(columns=[fid])
    return gdf


def process_centerline_chunk(chunk_data, footprint_gdf, footprint_sindex, unique_id_length):
    """
    Process a chunk of centerlines using spatial index for faster matching.
    This function is designed to be run in parallel.
    """
    chunk_indices, chunk_geometries = chunk_data
    results = []

    for idx, centerline in zip(chunk_indices, chunk_geometries):
        if centerline is None or centerline.is_empty:
            results.append((idx, None))
            continue

        # Generate sample points along the centerline
        points = [centerline.interpolate(i / 10, normalized=True) for i in range(11)]

        # Get bounding box of all points to query spatial index
        all_points = [Point(p.x, p.y) for p in points]
        minx = min(p.x for p in all_points)
        miny = min(p.y for p in all_points)
        maxx = max(p.x for p in all_points)
        maxy = max(p.y for p in all_points)

        # Query spatial index for candidate polygons
        possible_matches_idx = list(footprint_sindex.intersection((minx, miny, maxx, maxy)))

        if not possible_matches_idx:
            results.append((idx, None))
            continue

        # Only check the candidate polygons
        candidate_polygons = footprint_gdf.iloc[possible_matches_idx]

        # Count points in each candidate polygon
        point_counts = {}
        for poly_idx in possible_matches_idx:
            poly = footprint_gdf.iloc[poly_idx].geometry
            count = sum(poly.contains(point) for point in points)
            if count >= 5:  # Only store if meets threshold
                point_counts[poly_idx] = count

        if point_counts:
            # Get the polygon with the most points
            best_match_idx = max(point_counts.items(), key=lambda x: x[1])[0]
            unique_id = footprint_gdf.iloc[best_match_idx]['UniqueID']
            results.append((idx, unique_id))
        else:
            results.append((idx, None))

    return results


def assign_unique_ids_with_points_parallel(footprint_gdf, centerline_gdf, debug=False,
                                           unique_id_length=8, n_workers=None):
    """
    Optimized version using spatial index and multiprocessing.
    """
    # Fix invalid geometries for both layers
    logging.info("Fixing invalid geometries...")
    footprint_gdf["geometry"] = footprint_gdf["geometry"].apply(fix_invalid_geometry)
    centerline_gdf["geometry"] = centerline_gdf["geometry"].apply(fix_invalid_geometry)

    # Drop features with invalid geometries
    footprint_gdf = footprint_gdf[footprint_gdf["geometry"].notna()].copy()
    centerline_gdf = centerline_gdf[centerline_gdf["geometry"].notna()].copy()

    # Ensure footprint polygons have a UniqueID (assign if missing)
    if "UniqueID" not in footprint_gdf.columns:
        footprint_gdf["UniqueID"] = footprint_gdf.index.map(lambda _: generate_random_id(unique_id_length))

    # Prepare the centerline UniqueID column
    if "UniqueID" not in centerline_gdf.columns:
        centerline_gdf["UniqueID"] = None

    # Create spatial index for footprints
    logging.info("Building spatial index for footprints...")
    footprint_sindex = footprint_gdf.sindex

    # Prepare data for multiprocessing
    centerline_indices = centerline_gdf.index.tolist()
    centerline_geometries = centerline_gdf.geometry.tolist()

    # Determine number of workers
    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # Cap at 8 workers

    # Split data into chunks
    chunk_size = max(1, len(centerline_indices) // (n_workers * 4))  # More chunks than workers
    chunks = []

    for i in range(0, len(centerline_indices), chunk_size):
        chunk_indices = centerline_indices[i:i + chunk_size]
        chunk_geoms = centerline_geometries[i:i + chunk_size]
        chunks.append((chunk_indices, chunk_geoms))

    logging.info(f"Processing {len(centerline_gdf)} centerlines using {n_workers} workers...")

    # Create partial function with fixed arguments
    process_func = partial(process_centerline_chunk,
                           footprint_gdf=footprint_gdf,
                           footprint_sindex=footprint_sindex,
                           unique_id_length=unique_id_length)

    # Process chunks in parallel
    with Pool(n_workers) as pool:
        chunk_results = pool.map(process_func, chunks)

    # Combine results and update centerline_gdf
    matched_count = 0
    for chunk_result in chunk_results:
        for idx, unique_id in chunk_result:
            if unique_id is not None:
                centerline_gdf.at[idx, "UniqueID"] = unique_id
                matched_count += 1

    logging.info(f"Matched {matched_count} out of {len(centerline_gdf)} centerlines "
                 f"({matched_count / len(centerline_gdf) * 100:.1f}%)")

    return footprint_gdf, centerline_gdf


def assign_unique_ids_with_points(footprint_gdf, centerline_gdf, debug=False, unique_id_length=8):
    """
    Original function with spatial index optimization (single-threaded fallback).
    """
    # Fix invalid geometries for both layers
    footprint_gdf["geometry"] = footprint_gdf["geometry"].apply(fix_invalid_geometry)
    centerline_gdf["geometry"] = centerline_gdf["geometry"].apply(fix_invalid_geometry)

    # Drop features with invalid geometries
    footprint_gdf = footprint_gdf[footprint_gdf["geometry"].notna()].copy()
    centerline_gdf = centerline_gdf[centerline_gdf["geometry"].notna()].copy()

    # Ensure footprint polygons have a UniqueID (assign if missing)
    if "UniqueID" not in footprint_gdf.columns:
        footprint_gdf["UniqueID"] = footprint_gdf.index.map(lambda _: generate_random_id(unique_id_length))

    # Prepare the centerline UniqueID column
    if "UniqueID" not in centerline_gdf.columns:
        centerline_gdf["UniqueID"] = None

    # Create spatial index
    logging.info("Building spatial index...")
    footprint_sindex = footprint_gdf.sindex

    # Iterate over each centerline and match it to a footprint polygon
    total_lines = len(centerline_gdf)
    for i, (line_index, centerline_row) in enumerate(centerline_gdf.iterrows()):
        if i % 100 == 0:
            logging.info(f"Processing centerline {i}/{total_lines} ({i / total_lines * 100:.1f}%)")

        centerline = centerline_row.geometry
        if centerline is None or centerline.is_empty:
            logging.warning(f"Skipping invalid Centerline ID {line_index}.")
            continue

        # Generate 11 evenly spaced points along the centerline
        points = [centerline.interpolate(i / 10, normalized=True) for i in range(11)]

        # Get bounding box of centerline to query spatial index
        bounds = centerline.bounds

        # Query spatial index for candidate polygons
        possible_matches_idx = list(footprint_sindex.intersection(bounds))

        if not possible_matches_idx:
            logging.debug(f"No candidate polygons for Centerline ID {line_index}.")
            continue

        # Only check the candidate polygons (much faster!)
        candidate_polygons = footprint_gdf.iloc[possible_matches_idx]

        # Count how many points fall within each candidate polygon
        point_counts = candidate_polygons.geometry.apply(
            lambda poly: sum(poly.contains(point) for point in points)
        )

        # Identify polygons with at least 5 sample points inside
        matching_polygons = point_counts[point_counts >= 5]

        if not matching_polygons.empty:
            # Get the best match (most points)
            best_match_idx = matching_polygons.idxmax()
            matching_polygon_index = possible_matches_idx[
                candidate_polygons.index.get_loc(best_match_idx)
            ]
            unique_id = footprint_gdf.iloc[matching_polygon_index]["UniqueID"]
            centerline_gdf.at[line_index, "UniqueID"] = unique_id

            if debug:
                fig, ax = plt.subplots(figsize=(10, 10))
                footprint_gdf.plot(ax=ax, color="none", edgecolor="blue", label="Polygons")
                centerline_gdf.plot(ax=ax, color="grey", label="Centerlines")
                gpd.GeoDataFrame(geometry=[centerline], crs=centerline_gdf.crs).plot(
                    ax=ax, color="red", label="Selected Centerline"
                )
                gpd.GeoDataFrame(
                    geometry=[footprint_gdf.iloc[matching_polygon_index].geometry],
                    crs=footprint_gdf.crs
                ).plot(ax=ax, color="green", alpha=0.5, label="Matched Polygon")
                gpd.GeoDataFrame(geometry=points, crs=centerline_gdf.crs).plot(
                    ax=ax, color="yellow", label="Sampled Points", markersize=10
                )
                plt.legend()
                plt.title(f"Centerline ID {line_index} matched with Polygon ID {matching_polygon_index}")
                plt.show()

    return footprint_gdf, centerline_gdf


def update_path_with_suffix(input_path: str, suffix: str) -> str:
    """
    Update the input file path to include a suffix before the extension,
    saving the output in the same folder as the input.
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


def save_footprint(footprint_gdf, footprint_updated_path):
    """Save the updated footprint GeoDataFrame to the specified path."""
    try:
        footprint_gdf = drop_fid_column(footprint_gdf)
        footprint_gdf.to_file(footprint_updated_path, driver="GPKG", index=False)
        logging.info(f"Footprint saved successfully at: {footprint_updated_path}")
    except Exception as e:
        logging.error(f"Error saving Footprint GeoDataFrame: {e}")


def save_centerline(centerline_gdf, centerline_updated_path):
    """Save the updated centerline GeoDataFrame to the specified path."""
    try:
        centerline_gdf = drop_fid_column(centerline_gdf)
        centerline_gdf.to_file(centerline_updated_path, driver="GPKG", index=False)
        logging.info(f"Centerline saved successfully at: {centerline_updated_path}")
    except Exception as e:
        logging.error(f"Error saving Centerline GeoDataFrame: {e}")


def read_vector_file(path: str, layer: str = None) -> gpd.GeoDataFrame:
    """
    Reads a vector file. If the file is a GeoPackage and a layer is provided, it reads that layer.
    For shapefiles or other formats that don't support layers, it reads directly.
    """
    if path.startswith("file://"):
        path = path[7:]

    ext = os.path.splitext(path)[1].lower()

    if ext == ".gpkg":
        if layer and layer.lower() != 'default' and layer.lower() != 'null':
            return gpd.read_file(path, layer=layer)
        else:
            return gpd.read_file(path)
    else:
        # Do not pass 'layer' for shapefiles or other formats
        return gpd.read_file(path)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.logging.level)
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Use the assign_ID configuration subset
    assign_cfg = cfg.dataset
    footprint_path = assign_cfg.ground_footprint
    footprint_layer = assign_cfg.get("ground_footprint_layer", None)
    centerline_path = assign_cfg.centerline
    centerline_layer = assign_cfg.get("centerline_layer", None)
    unique_id_length = assign_cfg.unique_id_length
    debug_flag = assign_cfg.debug

    # Get number of workers from config or use default
    n_workers = assign_cfg.get("n_workers", None)
    use_parallel = assign_cfg.get("use_parallel", True)

    logging.info(f"Footprint Path: {footprint_path} | Layer: {footprint_layer}")
    logging.info(f"Centerline Path: {centerline_path} | Layer: {centerline_layer}")

    # Read files with optional layer specification
    footprint_gdf = read_vector_file(footprint_path, layer=footprint_layer)
    centerline_gdf = read_vector_file(centerline_path, layer=centerline_layer)

    # Perform matching so that each footprint polygon gets a consistent UniqueID,
    # and centerline features receive the matching UniqueID.
    logging.info("Matching centerlines to footprint polygons...")

    if use_parallel and len(centerline_gdf) > 100:  # Only use parallel for larger datasets
        footprint_gdf, centerline_gdf = assign_unique_ids_with_points_parallel(
            footprint_gdf, centerline_gdf, debug=debug_flag,
            unique_id_length=unique_id_length, n_workers=n_workers
        )
    else:
        footprint_gdf, centerline_gdf = assign_unique_ids_with_points(
            footprint_gdf, centerline_gdf, debug=debug_flag, unique_id_length=unique_id_length
        )

    # Save both footprint and centerline files separately in the same folders as their inputs.
    footprint_updated_path = update_path_with_suffix(footprint_path, "_footprint_ID")
    centerline_updated_path = update_path_with_suffix(centerline_path, "_centerline_ID")

    save_footprint(footprint_gdf, footprint_updated_path)
    save_centerline(centerline_gdf, centerline_updated_path)


if __name__ == "__main__":
    main()