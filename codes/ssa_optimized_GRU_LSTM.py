import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataset import create_sequences

print(" Initializing Final Optimized Pipeline on CPU...")


OPTIMAL_LSTM = {
    "lr": 0.00928,      
    "hidden": 16,     #UPDATE THE PARAMETERS AFTER SSA TUNING
    "layers": 1,      
    "batch": 16       
}

OPTIMAL_GRU = {
    "lr": 0.00900,      
    "hidden": 16,     
    "layers": 1,      
    "batch": 16       
}


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


DATA_ROOT = "data/processed"

PREPROCESSING_CONFIGS = ["sg_11_2"] 

MODELS_TO_TEST = ["LSTM", "GRU"]
EPOCHS = 15

if __name__ == "__main__":
    results = []

    for prep in PREPROCESSING_CONFIGS:
        data_path = os.path.join(DATA_ROOT, prep)
        if not os.path.exists(data_path): 
            print(f" Error: Could not find folder {data_path}")
            continue

        for model_name in MODELS_TO_TEST:
            print(f"\n Evaluating Optimized {model_name} on {prep}")
            
            cfg = OPTIMAL_LSTM if model_name == "LSTM" else OPTIMAL_GRU
            
            all_train_mse, all_test_mse, all_mae = [], [], []

            for file in os.listdir(data_path):
                if not file.endswith(".csv"): continue

                df = pd.read_csv(os.path.join(data_path, file))
                data = df[['Smooth_Close', 'Return', 'MA_10', 'EMA_10']].values
                X, y = create_sequences(data, 30)
                if len(X) < 50: continue

                split = int(0.8 * len(X))
                X_train_t = torch.tensor(X[:split], dtype=torch.float32)
                y_train_t = torch.tensor(y[:split], dtype=torch.float32).view(-1, 1)
                X_test_t = torch.tensor(X[split:], dtype=torch.float32)
                y_test_t = torch.tensor(y[split:], dtype=torch.float32).view(-1, 1)

                train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=cfg["batch"], shuffle=False)

                if model_name == "LSTM":
                    model = DynamicLSTM(4, cfg["hidden"], cfg["layers"])
                else:
                    model = DynamicGRU(4, cfg["hidden"], cfg["layers"])

                optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
                loss_fn = nn.MSELoss()

                model.train()
                for epoch in range(EPOCHS):
                    for batch_X, batch_y in train_loader:
                        pred = model(batch_X)
                        loss = loss_fn(pred, batch_y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    train_pred = model(X_train_t).numpy()
                    test_pred = model(X_test_t).numpy()

                all_train_mse.append(mean_squared_error(y_train_t.numpy(), train_pred))
                all_test_mse.append(mean_squared_error(y_test_t.numpy(), test_pred))
                all_mae.append(mean_absolute_error(y_test_t.numpy(), test_pred))

            if len(all_test_mse) > 0:
                results.append({
                    "model": model_name,
                    "preprocessing": prep,
                    "train_mse": np.mean(all_train_mse),
                    "test_mse": np.mean(all_test_mse),
                    "mae": np.mean(all_mae)
                })

    df_results = pd.DataFrame(results)
    df_results.to_csv("final_optimized_results.csv", index=False)
    print("\n Final Optimized Evaluation Completed!")