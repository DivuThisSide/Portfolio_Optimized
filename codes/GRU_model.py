import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataset import create_sequences

print("Initializing GRU Pipeline on CPU...")
# GRU MODEL ARCHITECTURE
class GRUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)



# EXPERIMENT HYPERPARAMETERS
DATA_ROOT = "data/processed"
SEQ_LENGTH = 30
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

PREPROCESSING_CONFIGS = [
    "sg_3_1",
    "sg_5_2",
    "sg_11_2",
    "no_sg"
]

# MAIN TRAINING LOOP
if __name__ == "__main__":
    results = []

    for prep in PREPROCESSING_CONFIGS:
        data_path = os.path.join(DATA_ROOT, prep)

        if not os.path.exists(data_path):
            print(f" Skipping {prep} (Directory not found)")
            continue

        print(f"\n Evaluating GRU on Preprocessing Configuration: {prep}")

        all_train_mse = []
        all_test_mse = []
        all_mae = []

        for file in os.listdir(data_path):
            if not file.endswith(".csv"):
                continue

            # 1. Load Data
            df = pd.read_csv(os.path.join(data_path, file))
            data = df[['Smooth_Close', 'Return', 'MA_10', 'EMA_10']].values

            # 2. Sequence Generation
            X, y = create_sequences(data, SEQ_LENGTH)

            if len(X) < 50:
                continue

            # 3. Train/Test Split
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # 4. Standard Tensor Conversion
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            # 5. Mini-Batch DataLoader Setup
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) 

            # 6. Model Initialization
            model = GRUModel(input_size=4)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            loss_fn = nn.MSELoss()

            # 7. Training Phase
            for epoch in range(EPOCHS):
                model.train()
                for batch_X, batch_y in train_loader:
                    pred = model(batch_X)
                    loss = loss_fn(pred, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # 8. Evaluation Phase
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor).numpy()
                test_pred = model(X_test_tensor).numpy()

            y_train_np = y_train_tensor.numpy()
            y_test_np = y_test_tensor.numpy()

            train_mse = mean_squared_error(y_train_np, train_pred)
            test_mse = mean_squared_error(y_test_np, test_pred)
            mae = mean_absolute_error(y_test_np, test_pred)

            all_train_mse.append(train_mse)
            all_test_mse.append(test_mse)
            all_mae.append(mae)

        if len(all_test_mse) == 0:
            continue

        # 9. Aggregate Portfolio Metrics
        avg_train = np.mean(all_train_mse)
        avg_test = np.mean(all_test_mse)
        avg_mae = np.mean(all_mae)

        print(f"   Train MSE : {avg_train:.6f}")
        print(f"   Test MSE  : {avg_test:.6f}")
        print(f"   MAE       : {avg_mae:.6f}")

        results.append({
            "preprocessing": prep,
            "train_mse": avg_train,
            "test_mse": avg_test,
            "mae": avg_mae
        })

    
    # SAVE PIPELINE RESULTS
    df_results = pd.DataFrame(results)
    df_results.to_csv("phase1_gru_results.csv", index=False)

    print("\n GRU Pipeline Execution Completed Successfully!")