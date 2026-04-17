import os
import pandas as pd
from preprocessing import preprocess_stock
from experiment_config import EXPERIMENTS

RAW_PATH = "data/raw"
BASE_OUTPUT = "data/processed"

# Filter for CSV files once before the loops
files = [f for f in os.listdir(RAW_PATH) if f.endswith(".csv")]

for exp in EXPERIMENTS:
    print(f"Running experiment: {exp['name']}")

    output_path = os.path.join(BASE_OUTPUT, exp["name"])
    os.makedirs(output_path, exist_ok=True)

    for file in files:
        df = pd.read_csv(os.path.join(RAW_PATH, file))
        
        # Apply experiment-specific preprocessing
        df = preprocess_stock(df, exp)

        save_path = os.path.join(output_path, file)
        df.to_csv(save_path, index=False)

print("All experiments completed.")