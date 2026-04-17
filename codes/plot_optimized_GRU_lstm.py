import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv("final_optimized_results.csv")
except FileNotFoundError:
    print(" Error: 'final_optimized_results.csv' not found. Make sure the evaluation script finished.")
    exit()

sns.set_theme(style="whitegrid")


fig, axes = plt.subplots(1, 3, figsize=(15, 6))


fig.suptitle('Optimized LSTM vs. GRU Performance (sg_11_2)', fontsize=16, fontweight='bold', y=1.05)

custom_palette = ['#4C72B0', '#DD8452']

# 1. Train MSE
sns.barplot(
    data=df, x='model', y='train_mse', 
    hue='model', legend=False, palette=custom_palette, ax=axes[0]
)
axes[0].set_title('Training Error (MSE)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Train MSE', fontsize=12)
axes[0].set_xlabel('Model Architecture', fontsize=12)

# 2. Test MSE 
sns.barplot(
    data=df, x='model', y='test_mse', 
    hue='model', legend=False, palette=custom_palette, ax=axes[1]
)
axes[1].set_title('Testing Error (True Generalization)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Test MSE', fontsize=12)
axes[1].set_xlabel('Model Architecture', fontsize=12)

# 3. MAE
sns.barplot(
    data=df, x='model', y='mae', 
    hue='model', legend=False, palette=custom_palette, ax=axes[2]
)
axes[2].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Test MAE', fontsize=12)
axes[2].set_xlabel('Model Architecture', fontsize=12)


plt.tight_layout()
plt.savefig("final_optimized_comparison.png", dpi=300, bbox_inches='tight')
print(" Plot saved as 'final_optimized_comparison.png'")
plt.show()