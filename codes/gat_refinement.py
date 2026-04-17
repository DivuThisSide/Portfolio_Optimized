import pandas as pd
import torch
from novel_portfolio_pipeline import OUT_DIR,HIDDEN
from stock_gat_model import StockGAT
from train_all_stocks import gat_order,adj,embeddings,metrics_rows
# GAT refinement + save metrics 
print("GAT refinement...")
ordered = [t for t in gat_order if t in embeddings]
if len(ordered) >= 2:
    emb = torch.stack([embeddings[t] for t in ordered])
    idx = [gat_order.index(t) for t in ordered]
    asub = adj[idx][:, idx].to(device)
    gat = StockGAT(in_dim=HIDDEN*2, hidden=32, out_dim=64,
                   n_heads=4, dropout=0.2).to(device)
    gat.eval()
    with torch.no_grad():
        ref = gat(emb.to(device), asub)
    torch.save({'tickers': ordered, 'embeddings': ref.cpu()},
               f"{OUT_DIR}/embeddings.pt")
    print(f"  Embeddings: {tuple(ref.shape)}")

mdf = pd.DataFrame(metrics_rows)
mdf.to_csv(f"{OUT_DIR}/metrics.csv", index=False)
mdf[['ticker','mu_hat','sigma_hat']].to_csv(f"{OUT_DIR}/uncertainty.csv", index=False)

print(f"\n{'='*55}")
print(f"Stocks trained : {len(mdf)}")
print(f"Avg RMSE       : {mdf['RMSE'].mean():.5f}")
print(f"Avg Dir-Acc    : {mdf['Dir_Acc'].mean():.3f}")
print(f"mu_hat range   : [{mdf['mu_hat'].min():+.5f}, {mdf['mu_hat'].max():+.5f}]")
if mdf['mu_hat'].std() < 0.0005:
    print(" WARNING: mu_hat has very low std — possible mode collapse.")
    print(" Check if 'Close' column is present in your CSVs.")
print(f"Outputs        : {OUT_DIR}/")
print(f"{'='*55}")