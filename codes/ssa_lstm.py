import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
import copy

from dataset import create_sequences

print("Initializing SSA Tuning on CPU...")


# DYNAMIC LSTM MODEL
class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# FITNESS FUNCTION 
def evaluate_fitness(params, X_train, y_train, X_val, y_val):
    """
    Decodes the sparrow's position into hyperparameters, trains the LSTM, 
    and returns the Validation MSE as the fitness score.
    """
    lr = params[0]
    hidden_size = int(round(params[1]))
    num_layers = int(round(params[2]))
    batch_size = int(round(params[3]))

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model = DynamicLSTM(input_size=4, hidden_size=hidden_size, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(5):  # Keep this low for tuning speed
        for batch_X, batch_y in train_loader:
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).numpy()
        val_mse = mean_squared_error(y_val.numpy(), val_pred)
    
    return val_mse


# THE SPARROW SEARCH ALGORITHM
def run_ssa(X_train, y_train, X_val, y_val, bounds, pop_size=10, max_iter=15):
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    PD = 0.2  
    SD = 0.1  
    ST = 0.8  
    
    p_num = int(pop_size * PD)
    s_num = int(pop_size * SD)

    X = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.zeros(pop_size)

    print("\nInitializing Swarm Fitness (This will take a moment)...")
    for i in range(pop_size):
        fitness[i] = evaluate_fitness(X[i], X_train, y_train, X_val, y_val)

    best_fitness = np.min(fitness)
    best_pos = X[np.argmin(fitness)].copy()

    for t in range(max_iter):
        sort_idx = np.argsort(fitness)
        X = X[sort_idx]
        fitness = fitness[sort_idx]
        
        worst_pos = X[-1].copy()
        best_pos_current = X[0].copy()

        # Update Producers
        R2 = np.random.rand()
        for i in range(p_num):
            if R2 < ST:
                X[i] = X[i] * np.exp(-i / (np.random.rand() * max_iter))
            else:
                X[i] = X[i] + np.random.randn() * np.ones(dim)

        # Update Scroungers
        for i in range(p_num, pop_size):
            if i > pop_size / 2:
                X[i] = np.random.randn() * np.exp((worst_pos - X[i]) / (i ** 2))
            else:
               
                A = np.random.choice([-1, 1], size=(1, dim))
                A_plus = A.T @ np.linalg.pinv(A @ A.T)
                # Flatten the result back to 1D to match the hyperparameter array
                X[i] = best_pos_current + np.abs(X[i] - best_pos_current) * A_plus.flatten()

        # Update Scouts
        scout_idx = np.random.choice(range(pop_size), s_num, replace=False)
        for i in scout_idx:
            if fitness[i] > best_fitness:
                X[i] = best_pos + np.random.randn() * np.abs(X[i] - best_pos)
            else:
                K = np.random.uniform(-1, 1)
                X[i] = X[i] + K * (np.abs(X[i] - worst_pos) / (fitness[i] - fitness[-1] + 1e-8))

        for i in range(pop_size):
            X[i] = np.clip(X[i], lb, ub)
            fitness[i] = evaluate_fitness(X[i], X_train, y_train, X_val, y_val)

        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_pos = X[np.argmin(fitness)].copy()

        print(f"Iteration {t+1}/{max_iter} | Best MSE: {best_fitness:.6f} | Params: LR={best_pos[0]:.4f}, Hidden={int(round(best_pos[1]))}, Layers={int(round(best_pos[2]))}, Batch={int(round(best_pos[3]))}")

    return best_pos, best_fitness


# MAIN EXECUTION
if __name__ == "__main__":
    BEST_PREP = "sg_11_2" 
    DATA_PATH = os.path.join("data/processed", BEST_PREP)
    
    PROXY_FILE = "RELIANCE.NS.csv" 
    file_path = os.path.join(DATA_PATH, PROXY_FILE)
    
    if not os.path.exists(file_path):
        print(f" Could not find proxy file at {file_path}.")
        exit()

    print(f" Loading Proxy Data for Tuning: {PROXY_FILE}")
    df = pd.read_csv(file_path)
    data = df[['Smooth_Close', 'Return', 'MA_10', 'EMA_10']].values
    
    X, y = create_sequences(data, seq_length=30)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    bounds = [
        [0.0001, 0.01],  # LR
        [16, 128],       # Hidden Size
        [1, 3],          # Num Layers
        [16, 128]        # Batch Size
    ]

    print("\n Starting Sparrow Search Algorithm (SSA)...")
    best_params, best_mse = run_ssa(
        X_train_t, y_train_t, X_val_t, y_val_t, 
        bounds=bounds, 
        pop_size=10,   
        max_iter=10    
    )
    print(" SSA TUNING COMPLETE")
    print(f"Minimum Validation MSE: {best_mse:.6f}")
    print("Optimal Hyperparameters Found:")
    print(f" - Learning Rate : {best_params[0]:.5f}")
    print(f" - Hidden Size   : {int(round(best_params[1]))}")
    print(f" - Num Layers    : {int(round(best_params[2]))}")
    print(f" - Batch Size    : {int(round(best_params[3]))}")