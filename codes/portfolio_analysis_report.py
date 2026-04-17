import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Initializing Supplementary Portfolio Analysis...")


# 1. DATA LOADING
SUMMARY_FILE = "mcvar_results/mcvar_summary.csv"

if not os.path.exists(SUMMARY_FILE):
    print(f"Error: Could not find {SUMMARY_FILE}. Please run mcvar_optimization.py first.")
    exit()

df = pd.read_csv(SUMMARY_FILE)


# 2. CONSOLE ANALYTICS & REPORTING


print(" AGGREGATE METRICS BY PORTFOLIO SIZE (K)")

# Grouping by K and formatting the output to be readable
k_group = df.groupby("k")[["mean_return", "sharpe", "sortino", "cumulative_return", "empirical_cvar"]].mean()
print(k_group.round(4).to_string())


print("  METHOD & SIZE SORTINO ANALYSIS")
# Grouping by Method and K specifically for Sortino
method_k_sortino = df.groupby(["method", "k"])["sortino"].mean().reset_index()
print(method_k_sortino.round(4).to_string(index=False))


print("ABSOLUTE BEST PORTFOLIO (SORTINO RATIO)")

# Finding the single best row based on the Sortino ratio
best = df.loc[df["sortino"].idxmax()]
print(f"Method        : {best['method'].upper()}")
print(f"Portfolio Size: Top {best['k']}")
print(f"Confidence    : {int(best['alpha']*100)}%")
print(f"Sortino Ratio : {best['sortino']:.4f}")
print(f"Sharpe Ratio  : {best['sharpe']:.4f}")
print(f"Mean Return   : {best['mean_return']:.6f}")
print(f"Volatility    : {best['volatility']:.6f}")
print(f"Weights File  : {best['weights_file']}")


# 3. VISUALIZATIONS

print("\n Generating Supplementary Plots...")
sns.set_theme(style="whitegrid")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Supplementary Risk & Sortino Analysis', fontsize=16, fontweight='bold', y=1.05)

# PLOT 1: Sortino Ratio vs K (Bar Chart) 
sortino_means = df.groupby("k")["sortino"].mean().reset_index()
sns.barplot(data=sortino_means, x="k", y="sortino", palette="viridis", ax=axes[0], hue="k", legend=False)

axes[0].set_title('Average Sortino Ratio by Portfolio Size (K)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Portfolio Size (K)', fontsize=12)
axes[0].set_ylabel('Mean Sortino Ratio', fontsize=12)

#  PLOT 2: Risk vs Return Scatter with Annotations
sns.scatterplot(
    data=df, 
    x="volatility", 
    y="mean_return", 
    hue="method", 
    style="method",
    s=150, # Marker size
    palette=['#4C72B0', '#DD8452'],
    ax=axes[1]
)


for i, row in df.iterrows():
    axes[1].text(
        row["volatility"] + 0.0001,
        row["mean_return"], 
        f"K={row['k']}", 
        fontsize=9, 
        alpha=0.8
    )

axes[1].set_title('Risk vs Return Landscape', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Risk (Volatility)', fontsize=12)
axes[1].set_ylabel('Mean Daily Return', fontsize=12)

plt.tight_layout()
save_path = "mcvar_results/extra_results_analysis.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f" Plots saved successfully to: {save_path}")
plt.show()