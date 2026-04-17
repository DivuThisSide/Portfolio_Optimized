import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(" Generating Final Portfolio Visualizations...")

# 1. Load the final optimization summary
summary_file = "mcvar_results/mcvar_summary.csv"
if not os.path.exists(summary_file):
    print(f"Error: Could not find {summary_file}. Run the optimizer first.")
    exit()

df = pd.read_csv(summary_file)

df_95 = df[df['alpha'] == 0.95].copy()

if df_95.empty:
    print(" No 95% confidence data found to plot.")
    exit()

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Final Portfolio Optimization Results (95% Confidence)', fontsize=18, fontweight='bold', y=1.05)


# CHART 1: Risk vs. Reward (Efficient Frontier Style)
sns.scatterplot(
    data=df_95, 
    x='volatility', 
    y='mean_return', 
    hue='method', 
    style='method',
    size='k', 
    sizes=(100, 400), 
    palette=['#4C72B0', '#DD8452'],
    ax=axes[0]
)

axes[0].set_title('Risk vs. Reward Landscape', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Risk (Daily Volatility)', fontsize=12)
axes[0].set_ylabel('Reward (Mean Daily Return)', fontsize=12)


# CHART 2: Sharpe Ratio Comparison
sns.barplot(
    data=df_95, 
    x='k', 
    y='sharpe', 
    hue='method', 
    palette=['#4C72B0', '#DD8452'],
    ax=axes[1]
)

axes[1].set_title('Risk-Adjusted Performance (Sharpe Ratio)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Portfolio Size (Top K Stocks)', fontsize=12)
axes[1].set_ylabel('Sharpe Ratio (Higher is Better)', fontsize=12)

# Clean up layout and save
plt.tight_layout()
save_path = "mcvar_results/final_portfolio_performance.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"Master plot saved successfully to: {save_path}")
plt.show()