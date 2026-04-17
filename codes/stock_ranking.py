import os
import pandas as pd
import numpy as np

INPUT_FILE = "stock_prediction_summary.csv"
OUTPUT_DIR = "screening_results"

# Generates Top 5, Top 10, and Top 15 lists
K_VALUES = [5, 10, 15]

# Professional Heuristic Quant Weights
W_RETURN = 0.40   # 40% Weight: High predicted return
W_ACCURACY = 0.30 # 30% Weight: Model has a high historical Hit Rate for this stock
W_TREND = 0.20    # 20% Weight: Stock has strong recent 5-day momentum
W_ERROR = 0.10    # 10% Penalty: Model has a high absolute error (MSE) for this stock



# HELPERS
def robust_normalize(series: pd.Series) -> pd.Series:
    """
    Normalizes data between 0 and 1, but clips extreme outliers 
    (bottom 5% and top 5%) to prevent them from breaking the distribution.
    """
    lower = series.quantile(0.05)
    upper = series.quantile(0.95)
    clipped = series.clip(lower, upper)
    
    denom = clipped.max() - clipped.min()
    if denom == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (clipped - clipped.min()) / denom


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = ["stock", "pred_return", "mse", "accuracy", "trend"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f" Missing columns in input file: {missing}. Make sure to run portfolio_screener.py first!")

    # Robust Normalization
    df["norm_return"] = robust_normalize(df["pred_return"])
    df["norm_accuracy"] = robust_normalize(df["accuracy"])
    df["norm_trend"] = robust_normalize(df["trend"])
    df["norm_mse"] = robust_normalize(df["mse"])

    return df


def score_equal_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["score_equal"] = (
        df["norm_return"] +
        df["norm_accuracy"] +
        df["norm_trend"] +
        (1 - df["norm_mse"])
    ) / 4.0

    df = df.sort_values("score_equal", ascending=False).reset_index(drop=True)
    return df


def score_heuristic_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Return is prioritized, momentum/accuracy add confidence, error reduces the score
    df["score_heuristic"] = (
        W_RETURN * df["norm_return"] +
        W_ACCURACY * df["norm_accuracy"] +
        W_TREND * df["norm_trend"] -
        W_ERROR * df["norm_mse"]
    )

    df = df.sort_values("score_heuristic", ascending=False).reset_index(drop=True)
    return df


def save_top_k(df_ranked: pd.DataFrame, k: int, method_name: str):
    topk = df_ranked.head(k).copy()
    out_path = os.path.join(OUTPUT_DIR, f"{method_name}_top_{k}.csv")
    topk.to_csv(out_path, index=False)
    print(f" Saved: {out_path}")
    print(topk[["stock", f"score_{method_name}", "pred_return", "accuracy", "trend", "mse"]].to_string(index=False))
    print("-" * 75)



# MAIN
def main():
    print("Initializing Final Portfolio Ranking Engine...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f" Error: '{INPUT_FILE}' not found. Please run portfolio_screener.py to generate it.")
        return

    df = prepare_features(df)

    # Apply Ranking Algorithms
    eq_ranked = score_equal_weights(df)
    eq_ranked.to_csv(os.path.join(OUTPUT_DIR, "ranked_equal_weights.csv"), index=False)

    h_ranked = score_heuristic_weights(df)
    h_ranked.to_csv(os.path.join(OUTPUT_DIR, "ranked_heuristic_weights.csv"), index=False)

    print("  EQUAL WEIGHTS RANKING (Balanced Portfolio)")
    for k in K_VALUES:
        save_top_k(eq_ranked, k, "equal")

    print(" HEURISTIC WEIGHTS RANKING (Aggressive Portfolio)")
    for k in K_VALUES:
        save_top_k(h_ranked, k, "heuristic")
        
    print("\n Full Pipeline Execution Complete! Check the /screening_results folder.")


if __name__ == "__main__":
    main()