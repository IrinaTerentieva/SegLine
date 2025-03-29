import os
import geopandas as gpd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from shapely import wkt
import pandas as pd


def init_worker(shared_polygon):
    """
    Initialize the worker with the shared merged polygon.
    """
    global merged_polygon
    merged_polygon = shared_polygon


def calculate_intersection(geom):
    """
    Calculate the intersection area and percentage for a single geometry.

    Args:
        geom (shapely.geometry): The bounding box geometry.

    Returns:
        dict: Intersection area and percentage or NaN if no overlap.
    """
    if merged_polygon.is_empty or geom.is_empty:
        return {
            'merged_intersection_area': np.nan,
            'merged_intersection_percentage': np.nan,
        }

    # Check for bounding box overlap using the geometries directly
    if not merged_polygon.intersects(geom):
        return {
            'merged_intersection_area': np.nan,
            'merged_intersection_percentage': np.nan,
        }

    # Calculate the intersection area
    intersection = geom.intersection(merged_polygon)
    intersection_area = intersection.area

    return {
        'merged_intersection_area': intersection_area,
        'merged_intersection_percentage': intersection_area / geom.area if geom.area > 0 else 0,
    }


def assign_seedling_to_largest_overlap(seedling_geom, polygon_gdf):
    """
    Assign a seedling to the plot with the largest intersection area.

    Args:
        seedling_geom (shapely.geometry): Geometry of the seedling.
        polygon_gdf (GeoDataFrame): GeoDataFrame containing polygon plots.

    Returns:
        int or None: The plot_id of the polygon with the largest overlap or None if no intersection.
    """
    intersecting_polygons = polygon_gdf[polygon_gdf.geometry.intersects(seedling_geom)]

    if intersecting_polygons.empty:
        return None

    intersecting_polygons = intersecting_polygons.copy()
    intersecting_polygons['intersection_area'] = intersecting_polygons.geometry.apply(
        lambda poly: seedling_geom.intersection(poly).area
    )

    largest_overlap = intersecting_polygons.loc[intersecting_polygons['intersection_area'].idxmax()]
    return largest_overlap['plot_id']


def assign_plot_ids_parallel(seedlings_gdf, polygon_gdf):
    """
    Assign plot IDs for seedlings in parallel.

    Args:
        seedlings_gdf (GeoDataFrame): GeoDataFrame containing seedlings.
        polygon_gdf (GeoDataFrame): GeoDataFrame containing polygon plots.

    Returns:
        GeoDataFrame: Updated seedlings with plot IDs assigned.
    """
    seedlings_gdf['plot_id'] = seedlings_gdf.geometry.apply(
        lambda geom: assign_seedling_to_largest_overlap(geom, polygon_gdf)
    )
    return seedlings_gdf


def calculate_intersection_with_merged_polygon(seedlings_gdf, polygon_gdf, output_path, num_cores=None, chunksize=100):
    """
    Calculate intersection of seedlings with merged polygon and assign plot IDs.

    Args:
        seedlings_gdf (GeoDataFrame): GeoDataFrame containing seedlings.
        polygon_gdf (GeoDataFrame): GeoDataFrame containing polygon plots.
        output_path (str): Path to save the updated GeoPackage.
        num_cores (int): Number of CPU cores to use.
        chunksize (int): Number of tasks per worker.
    """
    print("Merging polygons...")
    global merged_polygon
    merged_polygon = polygon_gdf.geometry.unary_union

    print(f"Using {num_cores or cpu_count()} cores for parallel processing...")
    with Pool(processes=num_cores or cpu_count(), initializer=init_worker, initargs=(merged_polygon,)) as pool:
        results = list(
            tqdm(
                pool.imap(calculate_intersection, seedlings_gdf.geometry, chunksize=chunksize),
                total=len(seedlings_gdf),
                desc="Processing intersections"
            )
        )

    seedlings_gdf['merged_intersection_area'] = [res['merged_intersection_area'] for res in results]
    seedlings_gdf['merged_intersection_percentage'] = [res['merged_intersection_percentage'] for res in results]

    seedlings_gdf['locationBB'] = seedlings_gdf['merged_intersection_percentage'].apply(
        lambda perc: "inner" if perc > 0.9 else ("outer" if perc >= 0.5 else 'forest')
    )

    print("Assigning plots to seedlings...")
    seedling_chunks = np.array_split(seedlings_gdf, num_cores or cpu_count())

    with Pool(num_cores or cpu_count()) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    assign_plot_ids_parallel,
                    [(chunk, polygon_gdf) for chunk in seedling_chunks]
                ),
                total=num_cores or cpu_count(),
                desc="Assigning plot IDs"
            )
        )

    seedlings_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=seedlings_gdf.crs)

    print(f"Saving results to {output_path}...")
    seedlings_gdf.to_file(output_path, layer="seedlings_with_intersection", driver="GPKG")

def main():
    # File paths
    seedling_file = r"D:\Blueberry_NEW\7_products\PD5\seedlings\all_seedlings_combined_300mm_subset_crs.gpkg"
    polygon_path = r"D:\Blueberry_NEW\7_products\footprint\_combine_all\segmenting\16_trimmed_footprint_QC_plots20m2_noSlivers_newSegID_QC.gpkg"
    output_path = r"D:\Blueberry_NEW\7_products\PD5\seedlings\all_seedlings_combined_300mm_subset_crs_intersections.gpkg"

    seedlings_gdf = gpd.read_file(seedling_file)
    polygons_gdf = gpd.read_file(polygon_path)

    calculate_intersection_with_merged_polygon(seedlings_gdf, polygons_gdf, output_path, num_cores=cpu_count(), chunksize=100)

if __name__ == "__main__":
    main()