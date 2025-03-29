import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, GeometryCollection
from shapely.ops import split, linemerge
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import os


def extend_line(line, extension_distance=100):
    """
    Extend a line at both ends by a given distance.
    """
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

    # Create extended line
    extended_coords = [start_extension] + coords + [end_extension]
    return LineString(extended_coords)


import numpy as np
from shapely.geometry import LineString

def generate_perpendiculars(centerline, avg_width, target_area, max_splitter_length=10):
    """
    Generate perpendicular lines to the centerline at intervals calculated to achieve a target area.
    Debug spacing and generated perpendiculars.

    Args:
        centerline (LineString): The input centerline geometry.
        avg_width (float): Average width for calculating spacing.
        target_area (float): Target area to calculate spacing.
        extension_distance (float): Length of perpendicular lines on each side of the centerline.

    Returns:
        list: List of perpendicular LineString objects.
    """
    if avg_width <= 0 or target_area <= 0:
        raise ValueError("Average width and target area must be positive.")

    # Calculate spacing
    spacing = target_area / avg_width
    # print(f"Calculated spacing: {spacing}")
    if spacing <= 0 or centerline.length <= 0:
        print("Invalid spacing or centerline length.")
        return []

    perpendiculars = []
    for distance in np.arange(0, centerline.length, spacing):
        # Interpolate a point on the centerline
        point = centerline.interpolate(distance)
        next_point = centerline.interpolate(min(distance + 1, centerline.length))

        # # Debug: Show interpolated points
        # print(f"Interpolated point at distance {distance}")

        # Calculate perpendicular vector
        dx, dy = next_point.x - point.x, next_point.y - point.y
        perpendicular_vector = (-dy, dx)
        length = np.sqrt(perpendicular_vector[0]**2 + perpendicular_vector[1]**2)
        unit_vector = (perpendicular_vector[0] / length, perpendicular_vector[1] / length)
        # print('max_splitter_length', max_splitter_length)

        half_length = max_splitter_length / 2
        start = (point.x - unit_vector[0] * half_length,
                 point.y - unit_vector[1] * half_length)
        end = (point.x + unit_vector[0] * half_length,
               point.y + unit_vector[1] * half_length)
        perpendicular_line = LineString([start, end])

        # Debug: Show generated perpendicular line
        # print(f"Generated perpendicular line: {perpendicular_line}")

        perpendiculars.append(perpendicular_line)

    # Debug: Summary of generated perpendiculars
    # print(f"Total perpendiculars generated: {len(perpendiculars)}")

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
        print(f"Error splitting geometry: {e}")
        return []


def process_polygon(footprint_row, centerline_gdf, smooth_centerline_gdf, target_area, extension_distance=50):
    """
    Process a single polygon using centerline and smooth_centerline.
    """
    unique_id = footprint_row["UniqueID"]
    polygon = footprint_row.geometry
    avg_width = footprint_row.get("avg_width", 0)

    max_width = avg_width + 10

    if max_width <= 5:
        max_width = 15

    if avg_width >= 9:
        target_area = int(target_area * 2)

    # print('unique_id:', unique_id)
    # print('max_width:', max_width)

    matched_smooth_centerline = smooth_centerline_gdf[smooth_centerline_gdf["UniqueID"] == unique_id]
    matched_centerline = centerline_gdf[centerline_gdf["UniqueID"] == unique_id]

    if matched_smooth_centerline.empty or matched_centerline.empty:
        print(f"No centerlines for UniqueID: {unique_id}")
        return []

    smooth_centerline = matched_smooth_centerline.iloc[0].geometry
    centerline = matched_centerline.iloc[0].geometry

    if isinstance(smooth_centerline, MultiLineString):
        smooth_centerline = linemerge(smooth_centerline)
    if isinstance(centerline, MultiLineString):
        centerline = linemerge(centerline)

    extended_centerline = extend_line(centerline, extension_distance)
    extended_smooth_centerline = extend_line(smooth_centerline, extension_distance)
    try:
        perpendiculars = generate_perpendiculars(extended_smooth_centerline, avg_width, target_area, max_splitter_length=max_width)
    except Exception as e:
        perpendiculars = generate_perpendiculars(extended_centerline, avg_width, target_area, max_splitter_length=max_width)

    if len(perpendiculars) < 5:
        print('Bad perpendiculars, switch to regular centerline')
        perpendiculars = generate_perpendiculars(extended_centerline, avg_width, target_area, max_splitter_length=max_width)

    # plot_perpendiculars(extended_smooth_centerline, perpendiculars)

    # print('N of perpendiculars: ', len(perpendiculars))
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
    print(results)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_polygon, row, centerline_gdf, smooth_centerline_gdf, target_area)
            for _, row in footprint_gdf.iterrows()
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing polygons"):
            result = future.result()
            if result:
                results.extend(result)

    print('Saving:')
    # Create GeoDataFrame with all original columns from footprint_gdf
    split_polygons_gdf = gpd.GeoDataFrame(results, crs=footprint_gdf.crs)
    split_polygons_gdf.to_file(output_path, layer="split_polygons", driver="GPKG")
    print(f"Split polygons saved to: {output_path}")

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
    elif filename.endswith(".gpkg"):
        updated_filename = filename.replace(".gpkg", "_ID.gpkg")
    else:
        raise ValueError("Unsupported file format. Only '.shp' and '.gpkg' are supported.")
    return os.path.join(output_dir, updated_filename)



# Load configuration
def main(debug=False):
    config_path = "C:/Users/X/Documents/FalconAndSwift/BRFN/recovery_assessment/footprint/BRFN/test.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    output_dir = config['datasets']['output_dir']
    num_workers = config['parameters']['num_workers']

    # Paths and parameters from config
    footprint_path = config['datasets']['ground_footprint']
    footprint_path = os.path.join(output_dir, os.path.basename(footprint_path).replace("10_", "12_").replace("_ID.gpkg", f"_sides.gpkg"))

    centerline_path = config['datasets']['centerline']
    #centerline_path = centerline_path.replace(".shp", f"_ID.gpkg")
    #centerline_path = os.path.join(output_dir, os.path.basename(centerline_path))
    smooth_centerline_path = centerline_path.replace("_ID.gpkg", f"_ID_smooth.gpkg")

    plot_area = int(config['parameters']['plot_area'] * 2)

    output_path = os.path.join(output_dir, os.path.basename(footprint_path).replace("12_", "13_").replace(f"_sides.gpkg", f"_plots{plot_area}m2.gpkg"))

    # Read input data
    footprint_gdf = gpd.read_file(footprint_path)
    centerline_gdf = gpd.read_file(centerline_path)
    smooth_centerline_gdf = gpd.read_file(smooth_centerline_path)

    print(f'Splitting with target area {plot_area} m2')


    num_workers = 16

    # Process polygons
    process_polygons_parallel(footprint_gdf, centerline_gdf, smooth_centerline_gdf, plot_area, output_path, max_workers=num_workers)

if __name__ == "__main__":
    main()
