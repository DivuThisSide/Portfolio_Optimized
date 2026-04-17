#  CNN-BiLSTM + Temporal Attention + MC Dropout
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.score = nn.Linear(h, 1, bias=False)
    def forward(self, hs):
        alpha = F.softmax(self.score(hs).squeeze(-1), dim=-1)
        return (alpha.unsqueeze(-1) * hs).sum(1), alpha

class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2,
                 dropout=0.3, conv_filters=64, conv_kernel=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_filters,
                      kernel_size=conv_kernel, padding=conv_kernel//2),
            nn.BatchNorm1d(conv_filters), nn.GELU(), nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(conv_filters, hidden_size, num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0.)
        self.attention  = TemporalAttention(hidden_size * 2)
        self.mc_dropout = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        xc = self.conv(x.permute(0,2,1)).permute(0,2,1)
        lo, _ = self.bilstm(xc)
        ctx, attn = self.attention(lo)
        return self.fc(self.mc_dropout(ctx)), attn

    def mc_predict(self, x, n_passes=50):
        self.train()
        with torch.no_grad():
            preds = torch.stack([self(x)[0].squeeze(-1)
                                 for _ in range(n_passes)], 0)
        self.eval()
        return preds.mean(0).cpu().numpy(), preds.std(0).cpu().numpy()

print("cnn_bilstm_attention_model defined ")