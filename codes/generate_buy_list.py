import os
import pandas as pd

print("Initializing Executive Portfolio Manager...\n")

SUMMARY_FILE = "mcvar_results/mcvar_summary.csv"

def generate_buy_list():
 
    if not os.path.exists(SUMMARY_FILE):
        print(f" Error: {SUMMARY_FILE} not found. Please run your pipeline first.")
        return

    
    df_summary = pd.read_csv(SUMMARY_FILE)
    
    if df_summary.empty:
        print("Error: The summary file is empty.")
        return

    #  Find the Absolute Best Portfolio (Highest Sharpe Ratio)
    best_portfolio = df_summary.loc[df_summary['sharpe'].idxmax()]
    
    weights_file = best_portfolio['weights_file']
    
    if not os.path.exists(weights_file):
        print(f"Error: Could not find the allocation file {weights_file}.")
        return

    #  Load the final target allocations
    df_weights = pd.read_csv(weights_file)
    
    df_buy_list = df_weights[df_weights['weight'] > 0.001].copy()
    

    df_buy_list['Allocation'] = (df_buy_list['weight'] * 100).map("{:.2f}%".format)

    
    
    print("  OPTIMAL QUANTITATIVE PORTFOLIO (BUY LIST)")
    print(f"Strategy Used  : {best_portfolio['method'].upper()}")
    print(f"Portfolio Size : Top {best_portfolio['k']} Stocks")
    print(f"Confidence (α) : {int(best_portfolio['alpha'] * 100)}%")
    print(f"Expected Daily Return : {best_portfolio['mean_return'] * 100:.3f}%")
    print(f"Max Tail Risk (CVaR)  : {best_portfolio['empirical_cvar'] * 100:.3f}%")
    print(f"Sharpe Ratio          : {best_portfolio['sharpe']:.3f}")
    print("\nTARGET ALLOCATIONS:")
    
   
    for index, row in df_buy_list.iterrows():
        ticker = str(row['stock']).ljust(15)
        allocation = str(row['Allocation']).rjust(10)
        print(f" {ticker} | {allocation}")
        
    print("\n Execution Complete. Good luck in the markets!")

if __name__ == "__main__":
    generate_buy_list()