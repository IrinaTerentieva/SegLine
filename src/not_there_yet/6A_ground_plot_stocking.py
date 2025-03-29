import geopandas as gpd

# File path to the plot information file
plot_info_path = "/media/irina/My Book/Blueberry/0_FieldData/blueberry_with_totals.gpkg"

# Read the GeoDataFrame
plot_info_gdf = gpd.read_file(plot_info_path)
print("Columns in input GeoDataFrame:", plot_info_gdf.columns)

# Rename `7_Site_Type` values in `metrics_gdf`
site_type_mapping = {
    "Upland": "xeric",
    "Upland Dry": "mesic",
    "Lowland Treed": "hydric",
    "Transitional": "mesic",
    None: "unknown"  # Handle None (missing values) as 'unknown'
}

# Apply the mapping
plot_info_gdf["ecosite"] = plot_info_gdf["7_Site_Type"].map(site_type_mapping)
# plot_info_gdf.loc[plot_info_gdf["1_Plot_ID"] == 4409, "ecosite"] = "hydric"

# Define column patterns for "above" and "below" groups across subplots
above_columns = [col for col in plot_info_gdf.columns if "__Above_" in col]
below_columns = [col for col in plot_info_gdf.columns if "__Below_" in col]

# Group by unique 1_Plot_ID
plot_groups = plot_info_gdf.groupby("1_Plot_ID")

# Initialize list for storing results
results = []

# Process each unique plot
for plot_id, group in plot_groups:
    # Combine geometries for the plot using unary_union
    plot_geometry = group.geometry.union_all()

    # Sum counts across all subplots in this plot for "above" and "below"
    total_above = group[above_columns].sum().sum()  # Total "above" seedlings
    total_below = group[below_columns].sum().sum()  # Total "below" seedlings

    # Count subplots with at least one "above" and "below"
    subplots_with_above = group[above_columns].gt(0).sum(axis=1).sum()  # Subplots with any "above" > 0
    subplots_with_below = group[below_columns].gt(0).sum(axis=1).sum()  # Subplots with any "below" > 0

    # Get the unique site type (assumes it's the same for all rows in the group)
    ecosite = group["ecosite"].iloc[0]

    # Determine stocking status
    if ecosite == "xeric":
        stocking = "pass" if subplots_with_above >= 5 else "fail"
    else:
        stocking = "pass" if subplots_with_above >= 7 else "fail"

    # Calculate density (per hectare, assuming 100m² plot area)
    plot_area_m2 = 100  # Assuming each plot is 100m²
    above_density = total_above * 10000 / plot_area_m2  # Convert to per hectare
    below_density = total_below * 10000 / plot_area_m2  # Convert to per hectare

    print(f"Plot ID: {plot_id}, Site Type: {ecosite}, Stocking: {stocking}, Subplots with Above: {subplots_with_above}")

    # Store the results for this plot
    results.append({
        "1_Plot_ID": plot_id,
        "ecosite": ecosite,
        "geometry": plot_geometry,
        "total_above": total_above,
        "total_below": total_below,
        "subplots_with_above": subplots_with_above,
        "subplots_with_below": subplots_with_below,
        "above_density": above_density,
        "below_density": below_density,
        "ground_stocking": stocking,  # Add stocking status
    })

# Convert results to a GeoDataFrame
results_gdf = gpd.GeoDataFrame(results, crs=plot_info_gdf.crs)

# Save results to a GeoPackage
output_path = "/media/irina/data/Blueberry/DATA/field/BRFN_Epicollect.gpkg"
results_gdf.to_file(output_path, driver="GPKG")
print(f"Aggregated results saved to {output_path}")
