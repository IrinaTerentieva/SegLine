import geopandas as gpd

points_path = "/media/irina/data/Blueberry/DATA/field/BRFN_Epicollect.gpkg"

PD = 5
seedlings = 'all'    ###'big'

metrics_path = f"file:///media/irina/data/Blueberry/DATA/status/all_sites_all_metrics_PD{PD}_v4.2_ass_{seedlings}seedlings_grID.gpkg"

output_path = metrics_path.replace('grID.gpkg', 'AA.gpkg')

# Step 1: Load both GeoPackages
metrics_gdf = gpd.read_file(metrics_path)
points_gdf = gpd.read_file(points_path)

metrics_gdf = metrics_gdf[~metrics_gdf["GroundID"].isna()]
print(metrics_gdf.columns)

# Ensure both GeoDataFrames have the same CRS
if metrics_gdf.crs != points_gdf.crs:
    points_gdf = points_gdf.to_crs(metrics_gdf.crs)


# Step 2: Calculate remote stocking metrics for each GroundID
def calculate_remote_stocking_per_groundid(metrics_gdf):
    """
    Calculate remote stocking metrics for each GroundID with additional logic based on segment_area.

    Args:
        metrics_gdf (GeoDataFrame): Input GeoDataFrame containing the data.

    Returns:
        dict: A dictionary containing remote stocking metrics for each GroundID.
    """
    # Group by GroundID and calculate metrics
    metrics_by_groundid = metrics_gdf.groupby("GroundID")

    results = {}
    for ground_id, group in metrics_by_groundid:
        # if ground_id != 4621.:
        #     continue
        print('\nGroundID: ', ground_id)
        # print(group[['GroundID', 'plot_id', 'plot_big_inner_count', 'plot_total_inner_count']])
        ecosite = group.ecosite.iloc[0]
        print('Ecosite: ', ecosite)

        # Get the segment area (assumes the same for all rows within a GroundID)
        segment_area = group["segment_area"].iloc[0] if "segment_area" in group.columns else None
        plot_area = group.head(1).geometry.iloc[0].area
        print('plot_area: ', plot_area)

        # Group by plot_id within this GroundID
        plots_with_inner = group.groupby("plot_id")["plot_big_inner_count"].sum()
        plots_with_outer = group.groupby("plot_id")["plot_total_inner_count"].sum()

        # Apply the logic based on segment_area
        if plot_area and plot_area < 15:
            # Sum plots with at least one seedling
            remote_inner_stocking = (plots_with_inner > 0).sum()
            remote_outer_stocking = (plots_with_outer > 0).sum()
        else:
            print('Wide!')
            # Use scaled logic: 0 for 0 seedlings, 1 for 1 seedling, 2 for 2+ seedlings
            remote_inner_stocking = plots_with_inner.apply(lambda x: 0 if x == 0 else 1 if x == 1 else 2).sum()
            remote_outer_stocking = plots_with_outer.apply(lambda x: 0 if x == 0 else 1 if x == 1 else 2).sum()

        print(
            f"Plot Area: {plot_area}, Remote BIG Inner Stocking: {remote_inner_stocking}, Remote ALL Inner Stocking: {remote_outer_stocking}")

        if ecosite == 'xeric':
            if remote_inner_stocking > 4:
                status = 'pass'
            else:
                status = 'fail'

        else:
            if remote_inner_stocking > 6:
                status = 'pass'
            else:
                status = 'fail'

        # Store the results
        results[ground_id] = {
            "remote_big_inner_stocking": remote_inner_stocking,
            "remote_total_inner_stocking": remote_outer_stocking,
            "segment_area": segment_area,
            'remote_status': status,
            'stocked': group['_stocked'],
        }
        print('Remote status {} for ecosite {}'.format(status, ecosite))
        print(remote_inner_stocking)
        print(remote_outer_stocking)

    return results


# Step 3: Get metrics for all GroundIDs
print("Calculating remote stocking metrics for each GroundID...")
remote_stocking_metrics = calculate_remote_stocking_per_groundid(metrics_gdf)

# Step 4: Assign metrics to points
print("Assigning metrics to points...")

def assign_metrics_to_points(points_gdf, remote_stocking_metrics, metrics_gdf):
    results = []
    for idx, point in points_gdf.iterrows():
        # print('\nPoint', point['1_Plot_ID'])
        # Find the closest GroundID
        distances = metrics_gdf.geometry.distance(point.geometry)
        closest_idx = distances.idxmin()
        closest_ground_id = metrics_gdf.loc[closest_idx, "GroundID"]
        # print('Ground ID: ', closest_ground_id)

        # Fetch metrics for the closest GroundID
        metrics = remote_stocking_metrics.get(closest_ground_id, {
            "remote_big_inner_stocking": 0,
            "remote_total_inner_stocking": 0,
            'remote_status': 0,
            "segment_area": None,
        })
        metrics["GroundID"] = closest_ground_id
        results.append(metrics)

    return results


# Assign metrics to points
assigned_metrics = assign_metrics_to_points(points_gdf, remote_stocking_metrics, metrics_gdf)

# Step 5: Convert results to DataFrame and merge with points
assigned_metrics_gdf = gpd.GeoDataFrame(assigned_metrics)  # Exclude geometry from metrics
points_with_metrics_gdf = points_gdf.reset_index(drop=True).join(assigned_metrics_gdf.reset_index(drop=True))

# Step 6: Save the updated GeoDataFrame
print(f"Saving updated GeoDataFrame to {output_path}...")
points_with_metrics_gdf.to_file(output_path, driver="GPKG")
print(f"Updated points saved to {output_path}")

print('\n****Checking')
print(points_with_metrics_gdf[points_with_metrics_gdf.GroundID == 4602.]['remote_big_inner_stocking'])
