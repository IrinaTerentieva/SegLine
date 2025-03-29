import os
import random
import string
import logging
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.errors import GEOSException
import hydra
from omegaconf import DictConfig, OmegaConf

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


def assign_unique_ids_with_points(footprint_gdf, centerline_gdf, debug=False, unique_id_length=8):
    """
    Use the centerline layer to match footprint polygons.
    For each centerline, sample points along its geometry and check which footprint polygon
    contains at least 5 of those points. The matching footprint polygon already has a UniqueID,
    which is then used to mark the centerline. This process ensures that each corresponding area
    (footprint polygon) receives a consistent UniqueID.

    Note: Although centerline matching is performed for validation purposes, only the footprint
    layer is output.
    """
    # Fix invalid geometries for both layers
    footprint_gdf["geometry"] = footprint_gdf["geometry"].apply(fix_invalid_geometry)
    centerline_gdf["geometry"] = centerline_gdf["geometry"].apply(fix_invalid_geometry)

    # Drop features with invalid geometries
    footprint_gdf = footprint_gdf[footprint_gdf["geometry"].notna()].copy()
    centerline_gdf = centerline_gdf[centerline_gdf["geometry"].notna()].copy()

    # Ensure footprint polygons have a UniqueID (assigned randomly if missing)
    if "UniqueID" not in footprint_gdf.columns:
        footprint_gdf["UniqueID"] = footprint_gdf.index.map(lambda _: generate_random_id(unique_id_length))

    # Initialize centerline UniqueID column for matching (not saved later)
    if "UniqueID" not in centerline_gdf.columns:
        centerline_gdf["UniqueID"] = None

    # Iterate over each centerline and match it to a footprint polygon
    for line_index, centerline_row in centerline_gdf.iterrows():
        centerline = centerline_row.geometry
        if centerline is None or centerline.is_empty:
            logging.warning(f"Skipping invalid Centerline ID {line_index}.")
            continue

        # Generate 11 evenly spaced points along the centerline
        points = [centerline.interpolate(i / 10, normalized=True) for i in range(11)]
        # Count how many points fall within each footprint polygon
        point_counts = footprint_gdf.geometry.apply(lambda poly: sum(poly.contains(point) for point in points))
        # Identify polygons with at least 5 sample points inside
        matching_polygons = point_counts[point_counts >= 5]

        if not matching_polygons.empty:
            matching_polygon_index = matching_polygons.index[0]
            unique_id = footprint_gdf.at[matching_polygon_index, "UniqueID"]
            centerline_gdf.at[line_index, "UniqueID"] = unique_id
            logging.info(
                f"Matched Centerline ID {line_index} with Polygon ID {matching_polygon_index} using UniqueID {unique_id}.")
            if debug:
                fig, ax = plt.subplots(figsize=(10, 10))
                footprint_gdf.plot(ax=ax, color="none", edgecolor="blue", label="Polygons")
                centerline_gdf.plot(ax=ax, color="grey", label="Centerlines")
                gpd.GeoDataFrame(geometry=[centerline], crs=centerline_gdf.crs).plot(ax=ax, color="red",
                                                                                     label="Selected Centerline")
                gpd.GeoDataFrame(geometry=[footprint_gdf.loc[matching_polygon_index].geometry],
                                 crs=footprint_gdf.crs).plot(ax=ax, color="green", alpha=0.5, label="Matched Polygon")
                gpd.GeoDataFrame(geometry=points, crs=centerline_gdf.crs).plot(ax=ax, color="yellow",
                                                                               label="Sampled Points", markersize=10)
                plt.legend()
                plt.title(f"Centerline ID {line_index} matched with Polygon ID {matching_polygon_index}")
                plt.show()
        else:
            logging.info(f"No matching polygon for Centerline ID {line_index}.")

    # Return the updated footprint (for output) and the centerline (for logging/debug purposes)
    return footprint_gdf, centerline_gdf


def update_path_with_id(input_path):
    """
    Update the input path to include '_ID' and save the updated file as a GeoPackage
    in the same directory as the original file.
    """
    # Remove 'file://' prefix if present
    if input_path.startswith("file://"):
        input_path = input_path[7:]
    dirname = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    if filename.endswith(".shp"):
        updated_filename = filename.replace(".shp", "_ID.gpkg")
    else:
        updated_filename = filename.replace(".gpkg", "_ID.gpkg")
    return os.path.join(dirname, updated_filename)


def save_footprint(footprint_gdf, footprint_updated_path):
    """Save the updated footprint GeoDataFrame to the specified path."""
    try:
        footprint_gdf = drop_fid_column(footprint_gdf)
        footprint_gdf.to_file(footprint_updated_path, driver="GPKG", index=False)
        logging.info(f"Footprint saved successfully at: {footprint_updated_path}")
    except Exception as e:
        logging.error(f"Error saving Footprint GeoDataFrame: {e}")


def read_vector_file(path, layer=None):
    """
    Reads a vector file. If a layer is provided (e.g. for a GeoPackage),
    it reads that specific layer; otherwise, it reads the file normally.
    """
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up logging level from config
    logging.basicConfig(level=cfg.logging.level)
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Use the assign_ID configuration subset
    assign_cfg = cfg.assign_ID
    footprint_path = assign_cfg.ground_footprint
    footprint_layer = assign_cfg.get("ground_footprint_layer", None)
    centerline_path = assign_cfg.centerline
    centerline_layer = assign_cfg.get("centerline_layer", None)
    unique_id_length = assign_cfg.unique_id_length
    debug_flag = assign_cfg.debug

    logging.info(f"Footprint Path: {footprint_path} | Layer: {footprint_layer}")
    logging.info(f"Centerline Path: {centerline_path} | Layer: {centerline_layer}")

    # Read files with optional layer specification
    footprint_gdf = read_vector_file(footprint_path, layer=footprint_layer)
    centerline_gdf = read_vector_file(centerline_path, layer=centerline_layer)

    # Perform matching so that each footprint polygon gets a consistent UniqueID.
    logging.info("Matching centerlines to footprint polygons...")
    footprint_gdf, _ = assign_unique_ids_with_points(
        footprint_gdf, centerline_gdf, debug=debug_flag, unique_id_length=unique_id_length
    )

    # Save only the footprint file (do not save the centerline layer) with UniqueIDs,
    # placing the output in the same directory as the original footprint.
    footprint_updated_path = update_path_with_id(footprint_path)
    save_footprint(footprint_gdf, footprint_updated_path)


if __name__ == "__main__":
    main()
