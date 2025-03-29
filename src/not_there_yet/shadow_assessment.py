import os
import numpy as np
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import geopandas as gpd
from multiprocessing import Pool, cpu_count
from rasterio.coords import disjoint_bounds
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import gc
import pandas as pd

def normalize_data(data, global_min, global_max):
    """
    Normalize data using global min and max values.
    """
    normalized = (data - global_min) / (global_max - global_min)  # Scale to 0-1
    normalized = np.clip(normalized, 0, 1)  # Clip values to 0-1 range
    return normalized * 255  # Scale to 0-255


def read_clipped_data(src, geom):
    """
    Reads and clips raster data to the exact geometry while ensuring correct nodata handling.
    """
    geom_mapping = [geom.__geo_interface__]

    if disjoint_bounds(src.bounds, geom.bounds):
        print("⚠️ Geometry does not overlap with raster bounds. Skipping...")
        return None, None

    try:
        clipped_data, clipped_transform = mask(src, geom_mapping, crop=True, nodata=255)

        # If the dataset has an alpha band, use it instead of the nodata attribute
        if "alpha" in src.descriptions:
            alpha_band = src.read_masks(1)  # Read the first band mask
            binary_mask = alpha_band > 0  # Convert mask to binary
        else:
            binary_mask = geometry_mask(
                [geom_mapping[0]],
                out_shape=clipped_data.shape[1:],  # Only height & width
                transform=clipped_transform,
                invert=True,
            )

        return clipped_data, binary_mask
    except rasterio.errors.RasterioIOError as e:
        print(f"❌ Error clipping data: {e}")
        return None, None



def calculate_global_min_max_random(optical_path, sample_fraction=0.01):
    """
    Calculate global min-max values for normalization using random patches of the raster.
    """
    global_min, global_max = float("inf"), float("-inf")

    with rasterio.open(optical_path) as src:
        block_windows = list(src.block_windows())
        sampled_windows = np.random.choice(len(block_windows), int(len(block_windows) * sample_fraction), replace=False)

        for idx in sampled_windows:
            _, window = block_windows[idx]
            block_data = src.read(window=window, masked=True)
            block_data = block_data[(block_data > 1) & (block_data < 255)]

            if block_data.size == 0:
                continue

            block_min = np.nanmin(block_data)
            block_max = np.nanmax(block_data)

            if np.isnan(block_min) or np.isnan(block_max):
                continue

            global_min = min(global_min, block_min)
            global_max = max(global_max, block_max)

    if global_min == float("inf") or global_max == float("-inf"):
        print("No valid data found in raster.")
        return None, None

    return global_min, global_max


def save_doi_debug_plot(output_path, doi_data, mask, segment_id, shadow_coverage, segment_area):
    """
    Save a debug plot of the DOI layer with annotations for shadow coverage and segment area.
    """
    os.makedirs(output_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    masked_doi = np.where(mask, doi_data, np.nan)
    im = ax.imshow(masked_doi, cmap="viridis")
    plt.colorbar(im, ax=ax, label="DOI Value")

    ax.set_title(f"Segment {segment_id} - Shadow Coverage: {shadow_coverage:.2f}%, Area: {segment_area:.2f} m²")
    ax.axis("off")

    debug_file = os.path.join(output_path, f"segment_{segment_id}_debug.png")
    plt.savefig(debug_file, dpi=300)
    plt.close(fig)


def calculate_doi(data):
    """
    Calculate DOI (Degree of Illumination) as the average of R, G, and B bands.
    """
    if data.shape[0] < 3:
        raise ValueError("Optical data must have at least 3 bands (R, G, B).")
    return np.mean(data[:3], axis=0)


def calculate_shadow_coverage(doi_data, mask, threshold=50):
    """
    Calculate the percentage of shadow coverage using DOI and a mask.
    """
    valid_pixels = doi_data[mask]
    shadow_pixels = valid_pixels < threshold

    total_inner_pixels = len(valid_pixels)
    if total_inner_pixels == 0:
        return 0

    shadow_coverage = (np.sum(shadow_pixels) / total_inner_pixels) * 100
    return shadow_coverage


def process_segment(unique_id, polygons_gdf, optical_src, global_min, global_max, output_path, threshold=50):
    """
    Process a single segment (UniqueID) to calculate shadow metrics using DOI.
    """

    # print('Processing shadows')
    segment_group = polygons_gdf[polygons_gdf["UniqueID"] == unique_id]
    results = []

    for segment_id, group in segment_group.groupby("SegmentID"):
        # print(f"Processing SegmentID: {segment_id}")
        combined_polygon = group.geometry.union_all()
        segment_area = combined_polygon.area
        # print(segment_area)

        if combined_polygon.is_empty or combined_polygon is None:
            print(f"Warning: Combined polygon is empty for segment {segment_id}.")
            continue

        optical_clipped, mask = read_clipped_data(optical_src, combined_polygon)
        if optical_clipped is None:
            print(f"Skipping SegmentID {segment_id} due to invalid optical data.")
            continue

        # try:
        normalized_data = normalize_data(optical_clipped, global_min, global_max)
        # print('Normalized')
        doi_data = calculate_doi(normalized_data)
        shadow_coverage = calculate_shadow_coverage(doi_data, mask, threshold)
        print('Shadows: ', shadow_coverage)

            # save_doi_debug_plot(os.path.join(output_path, "Figures"), doi_data, mask, segment_id, shadow_coverage, segment_area)

        original_attributes = group.iloc[0].to_dict()
        original_attributes.pop("geometry", None)

        results.append({
                **original_attributes,
                "geometry": combined_polygon,
                "segment_area": segment_area,
                "shadow_coverage": shadow_coverage,
            })
        # finally:
        #     del optical_clipped, mask, normalized_data, doi_data
        #     gc.collect()

    return results


def process_unique_ids(unique_ids, polygons_gdf, optical_path, output_path, global_min, global_max):
    """
    Multiprocessing helper to process a batch of UniqueIDs.
    """
    with rasterio.open(optical_path) as optical_src:
        results = []
        for unique_id in unique_ids:
            print(unique_id)
            results.extend(process_segment(unique_id, polygons_gdf, optical_src, global_min, global_max, output_path))
        return results


def process_chunks_with_parallel_unique_ids(footprint_path, optical_folder, output_path, batch_size=10):
    """
    Process all chunks sequentially but UniqueIDs within each chunk in parallel.
    """
    polygons_gdf = gpd.read_file(footprint_path)
    polygons_gdf = polygons_gdf[polygons_gdf.geometry.notnull()]
    polygons_gdf.set_geometry("geometry", inplace=True)
    polygons_gdf["geometry"] = polygons_gdf["geometry"].apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom
    )
    polygons_gdf["chunk"] = polygons_gdf["sitename"]

    region_to_site_mapping = {
        "GrizzlyCreekSite": "Grizzly Creek",
        "PA1-PinkMtnRanchRestored": "Pink Mountain",
        "PA2-E1(East)-AtticCreekRoadSite": "Attic Road",
        "PA2-E2(East)-BeattonRiver-SiteA": "Beatton River A",
        "PA2-E2(East)-BeattonRiver-SiteB": "Beatton River B",
        "PA2-W2(West)-RestoredWellpadAccess": "Wellpad Access",
        "PA2-W2(West)-SouthSikanniRoad": "South Sikanni"
    }

    # Apply the mapping correctly
    polygons_gdf["sitename"] = polygons_gdf["sitename"].replace(region_to_site_mapping)
    polygons_gdf["chunk"] = polygons_gdf["sitename"]  # Update chunk after mapping

    # Extract the first word from sitename to create 'chunk_short'
    polygons_gdf["chunk_short"] = polygons_gdf["sitename"].apply(lambda x: x.split(' ')[0])

    # Debug prints to verify
    print(polygons_gdf[["sitename", "chunk", "chunk_short"]].head(10))  # Check mapping

    chunk_names = polygons_gdf["chunk"].unique()  # Use updated chunk names
    chunk_names_short = polygons_gdf["chunk_short"].unique()  # Use short names
    print(f"Processing {len(chunk_names_short)} chunks...")  # Debug

    os.makedirs(output_path, exist_ok=True)

    available_rasters = {f for f in os.listdir(optical_folder) if f.endswith(".tif")}

    with tqdm(total=len(chunk_names), desc="Processing Chunks") as pbar:
        for chunk in chunk_names_short:
            matching_raster = next((r for r in available_rasters if chunk in r), None)

            if not matching_raster:
                print(f"No matching raster file for chunk: {chunk}")
                continue

            optical_path = os.path.join(optical_folder, matching_raster)
            print(f"Processing {chunk} -> Found raster: {optical_path}")

            if not os.path.exists(optical_path):
                print(f"Raster file does not exist for chunk: {chunk}")
                continue

            unique_ids = polygons_gdf[polygons_gdf["chunk_short"] == chunk]["UniqueID"].unique()
            global_min, global_max = calculate_global_min_max_random(optical_path)
            print(f"Global min {global_min} and max {global_max} for chunk {chunk}")

            # Split UniqueIDs into batches and process them in parallel
            unique_id_batches = [unique_ids[i:i + batch_size] for i in range(0, len(unique_ids), batch_size)]
            print('Unique batches: ', unique_id_batches)
            pool = Pool(cpu_count())

            results = []
            for batch_results in pool.starmap(
                process_unique_ids,
                [(batch, polygons_gdf, optical_path, output_path, global_min, global_max) for batch in unique_id_batches],
            ):
                results.extend(batch_results)

            pool.close()
            pool.join()

            save_results(results, os.path.join(output_path, f"{chunk}_shadows.gpkg"), polygons_gdf.crs)
            pbar.update()


def save_results(results, output_path, crs):
    """
    Save results to a GeoPackage.
    """
    print(f"Saving results to {output_path}")

    if results:
        # Convert results list into a GeoDataFrame
        gdf = gpd.GeoDataFrame(results, crs=crs)

        # Ensure shadow_coverage is a numeric column
        gdf["shadow_coverage"] = pd.to_numeric(gdf["shadow_coverage"], errors="coerce")

        # Ensure geometry column is properly set
        if "geometry" not in gdf.columns or gdf["geometry"].isnull().all():
            print("❌ Error: No valid geometry in results. Skipping save.")
            return

        # Save to GeoPackage
        gdf.to_file(output_path, driver="GPKG")
        print(f"✅ Results saved successfully to {output_path}")


def main():
    optical_folder = "/media/irina/My Book1/Blueberry/Orthos/5cm"

    footprint_path = f"/media/irina/data/Blueberry/DATA/metrics/all_sites_all_metrics_PD300_v4.1.gpkg"
    output_path = f"/media/irina/data/Blueberry/DATA/shadows/all_sites_all_metrics_PD300_shadows.gpkg"

    process_chunks_with_parallel_unique_ids(footprint_path, optical_folder, output_path)


if __name__ == "__main__":
    main()
