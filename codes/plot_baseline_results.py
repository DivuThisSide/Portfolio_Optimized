import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

try:
    df = pd.read_csv("baseline_experiment_results.csv")
except FileNotFoundError:
    print("Error: 'baseline_experiment_results.csv' not found.")
    sys.exit(1)

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot Training MSE
sns.barplot(
    data=df, 
    x='seq_len', 
    y='train_mse', 
    hue='architecture', 
    palette='Blues_d', 
    ax=axes[0]
)
axes[0].set_title('Training Error', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sequence Length (Days)', fontsize=12)
axes[0].set_ylabel('Train MSE', fontsize=12)
axes[0].legend(title='Hidden Layers')

# Plot Testing MSE
sns.barplot(
    data=df, 
    x='seq_len', 
    y='test_mse', 
    hue='architecture', 
    palette='Oranges_d', 
    ax=axes[1]
)
axes[1].set_title('Testing Error', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sequence Length (Days)', fontsize=12)
axes[1].set_ylabel('Test MSE', fontsize=12)
axes[1].legend(title='Hidden Layers')

plt.tight_layout()
plt.savefig("experiment_results_plot.png", dpi=300)
plt.show()