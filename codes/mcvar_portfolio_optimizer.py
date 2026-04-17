import os
import numpy as np
import pandas as pd
import cvxpy as cp

print("Initializing Institutional mCVaR Portfolio Optimizer...")

# 1. CONFIGURATION
RAW_DATA_DIR = "data/raw"           
SCREENING_DIR = "screening_results"  
OUTPUT_DIR = "mcvar_results"       

# Pipeline dimensions
METHODS = ["equal", "heuristic"]
K_VALUES = [5, 10, 15]

# Risk & Allocation Parameters
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
RETURN_FREQUENCY = "D"                
MIN_ENTROPY_FRACTION = 0.5             
TARGET_RETURN_MULTIPLIER = 0.8         # Target 80% of the GRU's predicted return
EPS = 1e-12                            # Small epsilon to prevent division by zero


# 2. DATA LOADING
def load_return_series(stock_name: str, freq: str = "D") -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, f"{stock_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing raw file for {stock_name}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    
    if 'close' not in df.columns:
        raise ValueError(f"Missing 'Close' column. Found: {list(df.columns)}")

    if df['close'].dtype == object:
        df['close'] = df['close'].astype(str).str.replace(',', '', regex=False)
    # Convert to float numbers. Anything broken becomes NaN.
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    date_col = None
    for col in ['date', 'timestamp', 'time', 'day']:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        # Standard chronological alignment using Dates
        df["Date"] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values("Date").dropna(subset=["close"]).copy()
        df = df.set_index("Date")
    else:
    
        df = df.dropna(subset=["close"]).copy()
        df = df.reset_index(drop=True)

    # Calculate returns
    if freq.upper() == "M" and date_col:
        df = df[["close"]].resample("M").last().dropna()
        returns = df["close"].pct_change().dropna()
    else:
        # If no dates exist, we fall back to row-by-row (daily) returns
        returns = df["close"].pct_change().dropna()

    return returns.rename(stock_name).to_frame()


def build_return_matrix(selected_df: pd.DataFrame, freq: str = "D"):
    stocks = selected_df["stock"].tolist()
    frames = []
    valid_stocks = []

    
    for stock in stocks:
        try:
            s = load_return_series(stock, freq=freq)
            frames.append(s)
            valid_stocks.append(stock)
        except Exception as e:
            print(f" Skipping {stock} due to raw data issue: {e}")

    if not frames:
        raise ValueError("All stocks failed to load. Check your data/raw/ folder.")

    # Merge valid stocks into a single aligned historical matrix
    merged = pd.concat(frames, axis=1, join="inner").dropna()
    merged = merged[valid_stocks]
    
    pred_mu = (
        selected_df.set_index("stock")
        .loc[valid_stocks, "pred_return"]
        .astype(float)
        .values
    )

    return merged, merged.values, pred_mu



# 3. MATHEMATICAL OPTIMIZATION
def solve_mcvar(R: np.ndarray, mu_pred: np.ndarray, alpha: float = 0.95, target_return: float | None = None, min_entropy_fraction: float | None = 0.5):
    T, n = R.shape
    if T < 10 or n < 2:
        raise ValueError("Not enough historical scenarios to optimize mCVaR.")

    if target_return is None:
        target_return = float(np.mean(mu_pred) * TARGET_RETURN_MULTIPLIER)

   
    base = abs(target_return) if abs(target_return) > EPS else 1e-4
    target_candidates = [
        target_return,
        target_return - 0.1 * base,
        target_return - 0.2 * base,
        target_return - 0.5 * base,
        target_return - 1.0 * base,
    ]

    entropy_max = np.log(n)
    if min_entropy_fraction is not None:
        entropy_candidates = [
            min_entropy_fraction * entropy_max,
            0.50 * min_entropy_fraction * entropy_max,
            None,
        ]
    else:
        entropy_candidates = [None]

    for ent_min in entropy_candidates:
        for tar in target_candidates:
            w = cp.Variable(n)
            tau = cp.Variable()
            u = cp.Variable(T)

            portfolio_return_scenarios = R @ w
            losses = -portfolio_return_scenarios

            # Objective: Minimize Conditional Value at Risk
            cvar = tau + (1.0 / ((1.0 - alpha) * T)) * cp.sum(u)

            constraints = [
                cp.sum(w) == 1,         
                w >= 0,                 
                u >= 0,
                u >= losses - tau,      
                mu_pred @ w >= tar,    
            ]

            if ent_min is not None:
                constraints.append(cp.sum(cp.entr(w)) >= ent_min)

            problem = cp.Problem(cp.Minimize(cvar), constraints)

            try:
                problem.solve(solver=cp.SCS, verbose=False, max_iters=30000)
            except Exception:
                continue

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w.value is not None:
                weights = np.array(w.value).reshape(-1)
                weights = np.clip(weights, 0, None)
                if weights.sum() > 0:
                    weights = weights / weights.sum() 
                return weights, float(cvar.value), float(tar), ent_min

    raise RuntimeError("mCVaR optimization mathematically failed for all targets.")



# 4. PORTFOLIO EVALUATION METRICS
def evaluate_portfolio(portfolio_returns: np.ndarray, weights: np.ndarray, alpha: float = 0.95):
    mean_ret = float(np.mean(portfolio_returns))
    vol = float(np.std(portfolio_returns, ddof=1) + EPS)

    downside = portfolio_returns[portfolio_returns < 0]
    downside_std = float(np.std(downside, ddof=1) + EPS) if len(downside) > 1 else EPS

    sharpe = mean_ret / vol
    sortino = mean_ret / downside_std

    losses = -portfolio_returns
    var_thr = np.quantile(losses, alpha)
    cvar_emp = float(np.mean(losses[losses >= var_thr])) if np.any(losses >= var_thr) else float(np.mean(losses))

    cum_return = float(np.prod(1.0 + portfolio_returns) - 1.0)
    entropy = float(-np.sum(weights * np.log(weights + EPS)))

    return {
        "mean_return": mean_ret,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "empirical_cvar": cvar_emp,
        "cumulative_return": cum_return,
        "entropy": entropy,
    }



# 5. MAIN EXECUTION LOOP
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_rows = []

    for method in METHODS:
        for k in K_VALUES:
            screen_file = os.path.join(SCREENING_DIR, f"{method}_top_{k}.csv")
            if not os.path.exists(screen_file):
                continue

            print(f"\n Optimizing {method.upper()} Portfolio (Top {k} Stocks)...")
            selected = pd.read_csv(screen_file)

            if selected.empty:
                continue

            try:
                merged_df, R, mu_pred = build_return_matrix(selected, freq=RETURN_FREQUENCY)
            except ValueError as e:
                print(e)
                continue
                
            base_target = float(np.mean(mu_pred) * TARGET_RETURN_MULTIPLIER)

            for alpha in CONFIDENCE_LEVELS:
                try:
                    weights, cvar_value, used_target, used_entropy_min = solve_mcvar(
                        R=R, mu_pred=mu_pred, alpha=alpha,
                        target_return=base_target, min_entropy_fraction=MIN_ENTROPY_FRACTION,
                    )
                except Exception as e:
                    print(f" Failed at {int(alpha*100)}% Confidence: {e}")
                    continue

                portfolio_returns = R @ weights

                weights_df = pd.DataFrame({
                    "stock": selected["stock"].tolist()[:len(weights)], 
                    "weight": weights
                }).sort_values("weight", ascending=False)

                weights_df["allocation_%"] = (weights_df["weight"] * 100).round(2)

                weights_path = os.path.join(OUTPUT_DIR, f"weights_{method}_k{k}_alpha{str(alpha).replace('.','')}.csv")
                weights_df.to_csv(weights_path, index=False)

                metrics = evaluate_portfolio(portfolio_returns, weights=weights, alpha=alpha)

                summary_rows.append({
                    "method": method,
                    "k": k,
                    "alpha": alpha,
                    "target_return": used_target,
                    "cvar_optimized": cvar_value,
                    "mean_return": metrics["mean_return"],
                    "volatility": metrics["volatility"],
                    "sharpe": metrics["sharpe"],
                    "sortino": metrics["sortino"],
                    "empirical_cvar": metrics["empirical_cvar"],
                    "cumulative_return": metrics["cumulative_return"],
                    "entropy": metrics["entropy"],
                    "weights_file": weights_path,
                })

                print(f" [Alpha {alpha}] Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | Cum_Ret: {metrics['cumulative_return']:.4f}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(OUTPUT_DIR, "mcvar_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n Optimization Complete! Master summary saved to: {summary_path}")
    else:
        print("No portfolios were mathematically viable.")


if __name__ == "__main__":
    run()