import geopandas as gpd



for PD in ['PD300', 'PD25', 'PD5']:

    # File paths
    ecosite_file = "/media/irina/data/Blueberry/DATA/ecosite/all_sites_all_metrics_PD300_ecoRF.gpkg"
    metrics_file = "/media/irina/data/Blueberry/DATA/metrics/all_sites_all_metrics_{}_v4.gpkg".format(PD)

    # Load the datasets
    print("Loading ecosite data...")
    ecosite_gdf = gpd.read_file(ecosite_file)[["plot_id", "predicted_landcover", "ecosite"]]

    print("Loading metrics data...")
    metrics_gdf = gpd.read_file(metrics_file)

    # Merge based on 'plot_id'
    print("Merging datasets...")
    merged_gdf = metrics_gdf.merge(ecosite_gdf, on="plot_id", how="left")

    # Save back to the metrics file
    output_file = metrics_file.replace(".gpkg", ".1.gpkg")  # Avoid overwriting original
    print(f"Saving merged data to {output_file}...")
    merged_gdf.to_file(output_file, driver="GPKG")

    print("Merge complete!")
