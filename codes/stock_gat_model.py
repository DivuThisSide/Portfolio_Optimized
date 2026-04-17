#  Graph Attention Network for inter-stock correlations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_f, out_f, n_heads=4, dropout=0.2, concat=True):
        super().__init__()
        self.n_heads, self.out_f, self.concat = n_heads, out_f, concat
        self.W = nn.Linear(in_f, out_f * n_heads, bias=False)
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_f))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky, self.drop = nn.LeakyReLU(0.2), nn.Dropout(dropout)

    def forward(self, x, adj):
        N  = x.size(0)
        Wx = self.W(x).view(N, self.n_heads, self.out_f)
        src = Wx.unsqueeze(1).expand(N, N, self.n_heads, self.out_f)
        dst = Wx.unsqueeze(0).expand(N, N, self.n_heads, self.out_f)
        e   = self.leaky((torch.cat([src, dst], -1) * self.a).sum(-1))
        e.masked_fill_((adj == 0).unsqueeze(-1).expand_as(e), float('-inf'))
        alpha = self.drop(F.softmax(e, dim=1))
        out   = (alpha.unsqueeze(-1) * Wx.unsqueeze(0)).sum(1)
        return out.reshape(N, self.n_heads * self.out_f) if self.concat \
               else out.mean(1)

class StockGAT(nn.Module):
    def __init__(self, in_dim, hidden=32, out_dim=64, n_heads=4, dropout=0.2):
        super().__init__()
        self.l1   = GATLayer(in_dim, hidden, n_heads, dropout, concat=True)
        self.l2   = GATLayer(hidden * n_heads, out_dim, 1, dropout, concat=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, adj):
        return self.l2(self.drop(F.elu(self.l1(x, adj))), adj)

def build_correlation_graph(returns_df, sector_map=None,
                             corr_threshold=0.4, sector_weight=0.5):
    corr = returns_df.corr(method='pearson').values.astype(np.float32)
    corr = np.nan_to_num(np.clip(corr, -1., 1.), nan=0.)
    adj  = np.where(np.abs(corr) >= corr_threshold, corr, 0.)
    if sector_map:
        tickers = list(returns_df.columns)
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if i != j and sector_map.get(ti) == sector_map.get(tj):
                    adj[i, j] = min(1., adj[i, j] + sector_weight)
    np.fill_diagonal(adj, 1.)
    adj = (adj + adj.T) / 2.
    return torch.FloatTensor(adj)

print("gat_model defined ")