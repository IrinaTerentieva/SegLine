import os
import glob
import yaml
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

def preprocess_chm(chm_path, seedlings_gdf):
    """
    Process a CHM raster by clamping values, masking seedlings, and exporting a new raster.
    """
    with rasterio.open(chm_path) as src:
        metadata = src.meta.copy()
        chm_data = src.read(1)
        transform = src.transform

        # Ensure CHM data is float32
        if not np.issubdtype(chm_data.dtype, np.floating):
            chm_data = chm_data.astype("float32")
            metadata["dtype"] = "float32"

        # Clamp values between 0 and 70
        #print(f"Processing {os.path.basename(chm_path)} - CHM range before clamping: {np.min(chm_data)}, {np.max(chm_data)}")
        #chm_data = np.clip(chm_data, 0, 70)
        #print(f"Processing {os.path.basename(chm_path)} - CHM range after clamping: {np.min(chm_data)}, {np.max(chm_data)}")

    # Ensure seedlings_gdf geometries are valid and aligned with CHM CRS
    if not seedlings_gdf.is_valid.all():
        seedlings_gdf = seedlings_gdf.buffer(0)
    if seedlings_gdf.crs != rasterio.open(chm_path).crs:
        seedlings_gdf = seedlings_gdf.to_crs(rasterio.open(chm_path).crs)

    # Create a seedling mask
    seedling_mask = geometry_mask(
        [geom.__geo_interface__ for geom in seedlings_gdf.geometry],
        transform=transform,
        invert=True,
        out_shape=chm_data.shape
    )

    # Debug: Check mask values
    print(f"Processing {os.path.basename(chm_path)} - Seedling mask values: {np.unique(seedling_mask)}")

    # Mask seedlings to -1
    chm_data[seedling_mask] = -1

    # Generate output filename with "_masked_seedlings" suffix
    output_path = os.path.splitext(chm_path)[0] + "_masked_seedlings.tif"

    # Update metadata
    metadata.update({"nodata": np.nan})
    with rasterio.open(output_path, "w", **metadata) as dst:
        dst.write(chm_data, 1)

    print(f"Processed {os.path.basename(chm_path)} -> Output saved at {output_path}")


def preprocess_trails(trails_path):
    """
    Preprocesses a .tif file by converting uint16 values to float32 and replacing nodata (0) with NaN.
    The processed file is saved with '_preprocessed' appended to the filename.

    :param trails_path: Path to the input .tif file.
    """
    # Open the raster file
    with rasterio.open(trails_path) as src:
        # Read the data
        data = src.read(1).astype(np.float32)  # Convert to float32

        # Define output filename
        dir_name, file_name = os.path.split(trails_path)
        base_name, ext = os.path.splitext(file_name)
        output_path = os.path.join(dir_name, f"{base_name}_preprocessed{ext}")

        # Define new metadata
        meta = src.meta.copy()
        meta.update({
            "dtype": "float32",
            "nodata": np.nan  # Set nodata value to NaN
        })

        # Save the new raster
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data, 1)

    print(f"Preprocessed file saved to: {output_path}")
    return output_path


def main():
    # Define input paths
    chm_folder = r"D:\Blueberry_NEW\7_products\PD25\chm"  # Change this to your CHM folder
    seedlings_path = r"D:\Blueberry_NEW\7_products\PD25\seedlings\all_seedlings_combined_50mm_subset_intersections_conf40.gpkg"
    trails_folder = r"D:\Blueberry_NEW\7_products\PD5\trails"

    # Load Seedlings GeoDataFrame
    #seedlings_gdf = gpd.read_file(seedlings_path)

    chm_files = sorted(glob.glob(os.path.join(chm_folder, "*_clamped.tif")))
    trails_files = sorted(glob.glob(os.path.join(trails_folder, "*.tif")))

    if not chm_files:
        print("No .tif files found in the specified folder.")
        return

    print(f"Found {len(chm_files)} CHM files to process...")
    print(f"Found {len(trails_files)} trails files to process...")
    # Process each CHM file
    # for chm_path in chm_files:
    #     preprocess_chm(chm_path, seedlings_gdf)
    #
    # print("All CHM files processed successfully!")

    # Process each trails file
    for trail_path in trails_files:
        preprocess_trails(trail_path)

    print('All trails processed successfully!')

if __name__ == "__main__":
    main()
