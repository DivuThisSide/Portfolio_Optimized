import pandas as pd
from portfolio_optimizer import w_nov,w_ew,w_mv,tickers_port
weight_data = []

for i in range(len(w_nov)):
    weight_data.append({
        'Ticker': tickers_port[i],  
        'Novel_mCVaR (%)': w_nov[i] * 100,
        'Equal_Weight (%)': w_ew[i] * 100,
        'Mean_Variance (%)': w_mv[i] * 100
    })

weights_df = pd.DataFrame(weight_data).set_index('Ticker')

print(f"  PORTFOLIO STOCK ALLOCATIONS (%)")
top_allocations = weights_df[weights_df['Novel_mCVaR (%)'] > 0.1].sort_values(by='Novel_mCVaR (%)', ascending=False)
print(top_allocations.to_string(formatters={
    'Novel_mCVaR (%)': '{:,.2f}%'.format,
    'Equal_Weight (%)': '{:,.2f}%'.format,
    'Mean_Variance (%)': '{:,.2f}%'.format
}))
print(f"Total Allocation Check: {w_nov.sum()*100:.1f}%")
