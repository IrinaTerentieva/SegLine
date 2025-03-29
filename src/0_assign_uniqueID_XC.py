import os
import geopandas as gpd
import random
import string
import yaml
from shapely.errors import GEOSException
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Set the seed for reproducibility
random.seed(101)

def generate_random_id(length=8):
    """Generate a random UniqueID consisting of letters and numbers."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def fix_invalid_geometry(geometry):
    """
    Attempt to fix invalid geometries using buffer(0) and handle errors gracefully.
    """
    try:
        if geometry is None or geometry.is_empty:
            return None
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Attempt to fix invalid geometries
        return geometry
    except GEOSException as e:
        print(f"[ERROR] Geometry error during fix: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error during fix: {e}")
        return None

def drop_fid_column(gdf):
    """
    Drop the 'fid' column from a GeoDataFrame if it exists.
    """
    if 'fid' in gdf.columns:
        print("Dropping 'fid' column to avoid schema conflict.")
        return gdf.drop(columns=['fid'])
    return gdf

def assign_unique_ids_with_points(footprint_gdf, centerline_gdf, debug=False):
    """
    Assign the same random UniqueID to polygons and centerlines based on points sampled along the centerlines.
    Fix invalid geometries before processing. Optionally visualize the matches for debugging.

    Args:
        footprint_gdf (GeoDataFrame): GeoDataFrame containing the polygons.
        centerline_gdf (GeoDataFrame): GeoDataFrame containing the centerlines.
        debug (bool): Whether to visualize the matches during assignment.

    Returns:
        (GeoDataFrame, GeoDataFrame): Updated GeoDataFrames with UniqueID column.
    """
    # Fix invalid geometries
    footprint_gdf["geometry"] = footprint_gdf["geometry"].apply(fix_invalid_geometry)
    centerline_gdf["geometry"] = centerline_gdf["geometry"].apply(fix_invalid_geometry)

    # Drop invalid geometries
    footprint_gdf = footprint_gdf[footprint_gdf["geometry"].notna()].copy()
    centerline_gdf = centerline_gdf[centerline_gdf["geometry"].notna()].copy()

    # Add UniqueID column if not present
    if "UniqueID" not in footprint_gdf.columns:
        footprint_gdf["UniqueID"] = footprint_gdf.index.map(lambda _: generate_random_id())
    if "UniqueID" not in centerline_gdf.columns:
        centerline_gdf["UniqueID"] = None

    for line_index, centerline_row in centerline_gdf.iterrows():
        centerline = centerline_row.geometry
        line_id = centerline_row.name  # Use index as fallback unique identifier

        # Skip invalid centerlines
        if centerline is None or centerline.is_empty:
            print(f"[WARNING] Skipping invalid Centerline ID {line_id}.")
            continue

        # Generate 10 points along the centerline
        points = [centerline.interpolate(i / 10, normalized=True) for i in range(11)]

        # Count how many points fall within each polygon
        point_counts = footprint_gdf.geometry.apply(lambda polygon: sum(polygon.contains(point) for point in points))

        # Find polygons with at least 5 points inside
        matching_polygons = point_counts[point_counts >= 5]

        if not matching_polygons.empty:
            # Assign the UniqueID of the first matching polygon to the centerline
            matching_polygon_index = matching_polygons.index[0]
            unique_id = footprint_gdf.at[matching_polygon_index, "UniqueID"]
            centerline_gdf.at[line_index, "UniqueID"] = unique_id

            print(f"Assigned UniqueID {unique_id} to Centerline ID {line_id} based on Polygon ID {matching_polygon_index}.")

            if debug:
                # Debugging visualization
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                footprint_gdf.plot(ax=ax, color="none", edgecolor="blue", label="Polygons")
                centerline_gdf.plot(ax=ax, color="grey", label="Centerlines")
                gpd.GeoDataFrame(geometry=[centerline], crs=centerline_gdf.crs).plot(
                    ax=ax, color="red", label="Selected Centerline"
                )
                gpd.GeoDataFrame(geometry=[footprint_gdf.loc[matching_polygon_index].geometry],
                                 crs=footprint_gdf.crs).plot(
                    ax=ax, color="green", alpha=0.5, label="Matched Polygon"
                )
                gpd.GeoDataFrame(geometry=points, crs=centerline_gdf.crs).plot(
                    ax=ax, color="yellow", label="Sampled Points", markersize=10
                )
                plt.legend()
                plt.title(f"Centerline ID {line_id} matched with Polygon ID {matching_polygon_index}")
                plt.show()
        else:
            print(f"No matching polygon for Centerline ID {line_id}.")

    return footprint_gdf, centerline_gdf


def save_geodataframes(footprint_gdf, centerline_gdf, footprint_updated_path, centerline_updated_path):
    """Save GeoDataFrames to specified paths."""
    try:
        footprint_gdf = drop_fid_column(footprint_gdf)
        centerline_gdf = drop_fid_column(centerline_gdf)

        footprint_gdf.to_file(footprint_updated_path, driver="GPKG", index=False)
        centerline_gdf.to_file(centerline_updated_path, driver="GPKG", index=False)
        print(f"GeoDataFrames saved successfully:\n- {footprint_updated_path}\n- {centerline_updated_path}")
    except Exception as e:
        print(f"Error saving GeoDataFrames: {e}")


def update_path_with_id(input_path, output_dir):
    """
    Update the input path to include '_ID' and ensure it is saved as a GeoPackage.

    Args:
        input_path (str): Path to the input file.
        output_dir (str): Directory where the updated file will be saved.

    Returns:
        str: Updated path with '_ID.gpkg'.
    """
    filename = os.path.basename(input_path)
    if filename.endswith(".shp"):
        updated_filename = filename.replace(".shp", "_ID.gpkg")
    else:
        updated_filename = filename.replace(".gpkg", "_ID.gpkg")
    return os.path.join(output_dir, updated_filename)


def main(debug=False):
    config_path = "C:/Users/X/Documents/FalconAndSwift/BRFN/recovery_assessment/footprint/BRFN/test.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    footprint_path = config["datasets"]["ground_footprint"]
    centerline_path = config["datasets"]["centerline"]
    output_dir = config["datasets"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Ground Footprint Path: {footprint_path}")
    print(f"Centerline Path: {centerline_path}")

    footprint_gdf = gpd.read_file(footprint_path)
    centerline_gdf = gpd.read_file(centerline_path)

    footprint_updated_path = update_path_with_id(footprint_path, output_dir)
    centerline_updated_path = update_path_with_id(centerline_path, output_dir)

    print("Assigning Unique IDs...")
    footprint_gdf, centerline_gdf = assign_unique_ids_with_points(footprint_gdf, centerline_gdf, debug=debug)

    save_geodataframes(footprint_gdf, centerline_gdf, footprint_updated_path, centerline_updated_path)


if __name__ == "__main__":
    main(debug=False)
