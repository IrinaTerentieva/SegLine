import os
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import substring
import math
import numpy as np
import yaml
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle (in degrees) between three points p1, p2, and p3.
    """
    dx1, dy1 = p2.x - p1.x, p2.y - p1.y
    dx2, dy2 = p3.x - p2.x, p3.y - p2.y
    dot_product = dx1 * dx2 + dy1 * dy2
    magnitude1 = math.sqrt(dx1**2 + dy1**2)
    magnitude2 = math.sqrt(dx2**2 + dy2**2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 180  # Treat as a straight line if any magnitude is zero
    angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
    return math.degrees(angle_rad)


def smooth_within_buffer(line, bad_point, buffer_distance):
    """
    Smooth a section of the line around a bad vertex (within buffer_distance).
    """
    try:
        buffer = bad_point.buffer(buffer_distance)
        line_within_buffer = line.intersection(buffer)

        if isinstance(line_within_buffer, LineString) and not line_within_buffer.is_empty:
            coords = list(line_within_buffer.coords)
            smoothed_coords = np.linspace(
                np.array(coords[0]),
                np.array(coords[-1]),
                len(coords),
            )
            smoothed_segment = LineString(smoothed_coords)

            start_proj = line.project(Point(smoothed_segment.coords[0]))
            end_proj = line.project(Point(smoothed_segment.coords[-1]))
            start_part = substring(line, 0, start_proj, normalized=False)
            end_part = substring(line, end_proj, line.length, normalized=False)

            return LineString(
                list(start_part.coords) + list(smoothed_segment.coords) + list(end_part.coords)
            )
    except Exception as e:
        print(f"Error smoothing buffer: {e}")
    return line


def smooth_centerline(line, angle_threshold=130, buffer_distance=2):
    """
    Smooth vertices of a LineString if their angles fall outside the acceptable range.
    """
    if not isinstance(line, LineString) or len(line.coords) < 3:
        return line

    coords = [Point(coord) for coord in line.coords]
    for i in range(1, len(coords) - 1):
        try:
            angle = calculate_angle(coords[i - 1], coords[i], coords[i + 1])
            if angle < angle_threshold or angle > (360 - angle_threshold):
                line = smooth_within_buffer(line, coords[i], buffer_distance)
        except Exception as e:
            print(f"Error processing angle at index {i}: {e}")
    return line


def process_centerline_line(line, angle_threshold, buffer_distance):
    """
    Process a single line for smoothing.
    """
    try:
        return smooth_centerline(line, angle_threshold, buffer_distance)
    except Exception as e:
        print(f"Error processing line: {e}")
        return line


def process_centerlines_worker(line, angle_threshold, buffer_distance):
    """
    Wrapper for passing arguments to `process_centerline_line` in multiprocessing.
    """
    return process_centerline_line(line, angle_threshold, buffer_distance)


def process_centerlines(centerline_gdf, angle_threshold=130, buffer_distance=2, num_workers=4):
    worker_function = partial(process_centerlines_worker, angle_threshold=angle_threshold, buffer_distance=buffer_distance)

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(worker_function, centerline_gdf["geometry"]),
                total=len(centerline_gdf),
                desc="Smoothing centerlines",
            )
        )
    centerline_gdf = centerline_gdf.copy()
    centerline_gdf["geometry"] = results
    return centerline_gdf


def main(debug=False):
    config_path = "C:/Users/X/Documents/FalconAndSwift/BRFN/recovery_assessment/footprint/BRFN/test.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    output_dir = config["datasets"]["output_dir"]

    centerline_path = config["datasets"]["centerline"]
    #centerline_path = centerline_path.replace(".shp", f"_ID.gpkg")
    #centerline_path = os.path.join(output_dir, os.path.basename(centerline_path))

    smoothening = config["parameters"]["smoothening"]

    centerline_gdf = gpd.read_file(centerline_path)
    print(centerline_gdf)


    print("Smoothing centerlines...")
    smoothed_gdf = process_centerlines(
        centerline_gdf, angle_threshold=130, buffer_distance=smoothening, num_workers=min(cpu_count(), 8)
    )

    if "fid" in smoothed_gdf.columns:
        smoothed_gdf = smoothed_gdf.drop(columns=["fid"])

    output_path = os.path.join(output_dir, os.path.basename(centerline_path).replace('.gpkg', '_smooth.gpkg'))
    try:
        smoothed_gdf.to_file(output_path, driver="GPKG")
        print(f"Smoothed centerlines saved to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()
