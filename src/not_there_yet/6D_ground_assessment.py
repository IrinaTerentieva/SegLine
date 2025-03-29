import geopandas as gpd
from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load data
# ground_path = "/media/irina/data/Blueberry/DATA/status/all_sites_all_metrics_PD5_v2eco_sha_assess_v1_AA.gpkg"
# ground_path = "/media/irina/data/Blueberry/DATA/status/all_sites_all_metrics_PD25_v2eco_sha_assess_v1_AA.gpkg"
# ground_path = "/media/irina/data/Blueberry/DATA/status/all_sites_all_metrics_PD300_v4.2_ass_allseedlings_AA.gpkg"

PD = 5
seedlings = 'all'    ###'big'

ground_path = f"/media/irina/data/Blueberry/DATA/status/all_sites_all_metrics_PD{PD}_v4.2_ass_{seedlings}seedlings_AA.gpkg"

if seedlings == 'all':
    type = 'remote_total_inner_stocking'
elif seedlings == 'big':
    type = 'remote_big_inner_stocking'

metrics_gdf = gpd.read_file(ground_path)
print(metrics_gdf.columns)
# print(metrics_gdf[["ecosite", '1_Plot_ID']] )

# Verify the changes
print("Updated Site Types:", metrics_gdf["ecosite"].unique())

output_dir = f"/media/irina/data/Blueberry/DATA/status/Charts/AA_PD{PD}_{seedlings}seedlings"
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

def save_figure(filename):
    """Save figure to the specified output directory."""
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()

def process_stocked_column(stocked):
    try:
        # Check if the value is a string
        if isinstance(stocked, str):
            # Extract the last value from each line and attempt to convert to an integer
            values = []
            for val in stocked.split("\n"):
                try:
                    # Split and extract the last token (assumed to be 0 or 1)
                    values.append(int(val.split()[-1]))
                except (ValueError, IndexError):
                    # Skip lines that can't be processed
                    continue
            # Aggregate: If any value is 1, return 1; otherwise, return 0
            return 1 if any(values) else 0
    except Exception as e:
        print(f"Error processing stocked value: {stocked}, Error: {e}")
    # Default to 0 if the value is invalid or missing
    return 0

# Apply processing to `_stocked` column
metrics_gdf["_stocked"] = metrics_gdf["stocked"].apply(process_stocked_column)

print("stocking (ground data) in metrics gdf:", metrics_gdf["ground_stocking"].unique())
print("stocked (remote assessment) in metrics gdf:", metrics_gdf["_stocked"].unique())

# Ensure required columns exist
required_columns = ["subplots_with_above", "remote_big_inner_stocking", "remote_total_inner_stocking", "GroundID", "segment_area"]
if not all(col in metrics_gdf.columns for col in required_columns):
    raise ValueError(f"Missing required columns in the dataset: {required_columns}")

def scatter_with_correlation_colored(x, y, site_types, xlabel, ylabel, title, colors):
    plt.figure(figsize=(8, 6))

    # Define plot limits
    min_val = min(x.min(), y.min()) - 1
    max_val = max(x.max(), y.max()) + 1

    # 1:1 Reference Line
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=2, label="1:1 Line")

    # Plot points with color based on site type
    for site_type, color in colors.items():
        subset = metrics_gdf[site_types == site_type]
        plt.scatter(
            subset[x.name], subset[y.name], alpha=0.7, s=80, edgecolors="black",
            label=f"{site_type.capitalize()} (n={len(subset)})", color=color
        )

    # Calculate and display Pearson correlation coefficient
    corr_coef, _ = pearsonr(x.dropna(), y.dropna())

    # Labels and title
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{title}\nPearson r = {corr_coef:.2f}", fontsize=14, fontweight="bold")

    # Improve aesthetics
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)  # Dashed grid lines
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()

    save_figure("scatter_correlation.png")



# Scatter plot for Ground vs Remote Inner Stocking
scatter_with_correlation_colored(
    metrics_gdf["subplots_with_above"],
    metrics_gdf[type],
    metrics_gdf["ecosite"],
    "Stocked Plots: Ground data",
    "Stocked Plots: Aerial data",
    "Ground vs Remote Stocking",
    colors={"mesic": "green", "xeric": "orange", "hydric": "blue"}
)


# 4. CONFUSION MATRIX

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(metrics_gdf):
    """
    Plots a confusion matrix for ground vs. remote stocking status with larger font sizes.
    """
    # Map "stocking" column to binary values: 'pass' -> 1, 'fail' -> 0
    metrics_gdf["ground_status_binary"] = metrics_gdf["ground_stocking"].map({"pass": 1, "fail": 0})
    metrics_gdf["remote_status_binary"] = metrics_gdf["remote_status"].map({"pass": 1, "fail": 0})

    # Calculate the confusion matrix
    cm = confusion_matrix(metrics_gdf["ground_status_binary"], metrics_gdf["remote_status_binary"], labels=[1, 0])
    cm_labels = ["Pass", "Fail"]

    # Plot the confusion matrix with larger font sizes
    plt.figure(figsize=(7, 6))  # Increase figure size
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels,
        annot_kws={"size": 18},  # Increase numbers inside the plot
        linewidths=0.5, linecolor="black",  # Gridlines for clarity
    )

    # Correct axis labels with larger font
    plt.xlabel("Remote Status", fontsize=18, fontweight="bold")
    plt.ylabel("Ground Status", fontsize=18, fontweight="bold")
    plt.title("High-Altitude Deployment", fontsize=20, fontweight="bold")

    # Increase tick labels size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Save and close
    save_figure("confusion_matrix.png")




metrics_gdf["ground_status_binary"] = metrics_gdf["ground_stocking"].map({"pass": 1, "fail": 0})
metrics_gdf["remote_status_binary"] = metrics_gdf["remote_status"].map({"pass": 1, "fail": 0})
plot_confusion_matrix(metrics_gdf)

# 3. Difference Metrics (Bias)
metrics_gdf["difference_inner"] = metrics_gdf[type] - metrics_gdf["subplots_with_above"]
metrics_gdf["difference_total"] = metrics_gdf[type] - metrics_gdf["subplots_with_above"]
print("Difference Metrics Summary:\n", metrics_gdf[["difference_inner", "difference_total"]].describe())

# 4. Error Metrics (RMSE, MAE)
rmse_inner = mean_squared_error(metrics_gdf["subplots_with_above"], metrics_gdf[type], squared=False)
mae_inner = mean_absolute_error(metrics_gdf["subplots_with_above"], metrics_gdf[type])
print(f"Inner BIG Stocking - RMSE: {rmse_inner:.2f}, MAE: {mae_inner:.2f}")

# 5. T-Test
t_stat_inner, p_val_inner = ttest_rel(metrics_gdf["subplots_with_above"], metrics_gdf[type])
print(f"T-Test Results (Inner Stocking): t-stat={t_stat_inner:.2f}, p-value={p_val_inner:.2e}")



def plot_difference_distribution(differences, title):
    plt.figure(figsize=(6, 6))

    # Determine x-axis range dynamically
    x_min, x_max = -10, 10

    # Plot histogram with KDE (Density Curve)
    sns.histplot(differences, kde=True, bins=20, color="royalblue", edgecolor="black", alpha=0.7)
    x_ticks = range(-10, 11, 1)  # Ticks from -10 to 10

    # Mean & Median Lines
    mean_val = differences.mean()
    median_val = differences.median()
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    plt.axvline(median_val, color="black", linestyle="-", linewidth=2, label=f"Median: {median_val:.2f}")

    # Title & Labels
    plt.title(f"Residual Distribution Plot", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Difference in Stocked Plots (Remote vs Ground Survey)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Aesthetic Improvements
    plt.xticks(ticks=x_ticks, fontsize=10)  # Set custom ticks
    plt.xlim(x_min, x_max)  # Dynamic x-axis limits
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=11, loc="upper right", frameon=True)
    plt.tight_layout()

    save_figure("residuals.png")

# Call the function
plot_difference_distribution(metrics_gdf["difference_inner"], "Residuals")

# 2. Bar Plots for Each GroundID
def plot_bars_by_groundid(metrics_gdf, stocking_type="remote_big_inner_stocking", rows=5, cols=6):
    unique_ground_ids = sorted(metrics_gdf["GroundID"].unique())
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for i, ground_id in enumerate(unique_ground_ids):
        if i >= len(axes):
            break
        ax = axes[i]
        group = metrics_gdf[metrics_gdf["GroundID"] == ground_id]
        ground_truth = group["subplots_with_above"].sum()
        remote_modeled = group[stocking_type].sum()
        ax.bar(["Ground Truth", "Remote Survey"], [ground_truth, remote_modeled], color=["blue", "orange"])
        ax.set_title(f"Ground Plot: {int(ground_id)}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(ground_truth, remote_modeled) + 5)

    # Hide unused subplots
    for j in range(len(unique_ground_ids), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_figure("bars_by_groundid.png")

plot_bars_by_groundid(metrics_gdf, stocking_type=type)

def scatter_and_residuals(metrics_gdf, type):
    # Create a figure with a specific width ratio: 60% (scatter) vs. 40% (residuals)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 2]})

    # Scatter Plot (60% width)
    ax1 = axes[0]
    colors = {"mesic": "green", "xeric": "orange", "hydric": "blue"}

    min_val = min(metrics_gdf["subplots_with_above"].min(), metrics_gdf[type].min()) - 1
    max_val = max(metrics_gdf["subplots_with_above"].max(), metrics_gdf[type].max()) + 1

    # Plot reference 1:1 line
    ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=2, label="1:1 Line")

    # Plot data points colored by `ecosite`
    for site_type, color in colors.items():
        subset = metrics_gdf[metrics_gdf["ecosite"] == site_type]
        ax1.scatter(
            subset["subplots_with_above"], subset[type], alpha=0.7, s=80, edgecolors="black",
            label=f"{site_type.capitalize()} (n={len(subset)})", color=color
        )

    # Calculate and display Pearson correlation coefficient
    corr_coef, _ = pearsonr(metrics_gdf["subplots_with_above"].dropna(), metrics_gdf[type].dropna())

    # Scatter plot styling
    ax1.set_xlabel("Stocked Plots: Ground Data", fontsize=12)
    ax1.set_ylabel("Stocked Plots: Aerial Data", fontsize=12)
    ax1.set_title(f"Ground vs Aerial Establishment Survey", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)

    # Residuals Distribution (40% width)
    ax2 = axes[1]
    x_min, x_max = -10, 10
    sns.histplot(metrics_gdf["difference_inner"], kde=True, bins=20, color="royalblue", edgecolor="black", alpha=0.7,
                 ax=ax2)

    # Mean & Median Lines
    mean_val = metrics_gdf["difference_inner"].mean()
    median_val = metrics_gdf["difference_inner"].median()
    ax2.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax2.axvline(median_val, color="black", linestyle="-", linewidth=2, label=f"Median: {median_val:.2f}")

    # Residuals plot styling
    ax2.set_title("Residual Distribution", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Aerial vs Ground Survey: Difference in Plot Stocking", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_xticks(range(-10, 11, 2))
    ax2.set_xlim(x_min, x_max)
    ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax2.legend(fontsize=11, loc="upper right", frameon=True)

    # Adjust layout
    plt.tight_layout()
    save_figure("scatter_and_residuals.png")


# Call the function to generate the combined figure
scatter_and_residuals(metrics_gdf, type)


import matplotlib.pyplot as plt

# Compute residuals
metrics_gdf["residuals"] = metrics_gdf[type] - metrics_gdf["subplots_with_above"]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=metrics_gdf["subplots_with_above"], y=metrics_gdf["residuals"], alpha=0.7, color="royalblue")

# Add reference line at 0 (ideal case)
plt.axhline(0, color="gray", linestyle="--", linewidth=2, label="Ideal (No Bias)")

# Labels and title
plt.xlabel("Ground Truth Stocking (%)", fontsize=12)
plt.ylabel("Residual (Remote - Ground)", fontsize=12)
plt.title("Residuals vs. Ground Stocking (%)", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

save_figure("residuals_vs_ground.png")
