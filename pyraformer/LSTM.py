from torch import nn
import torch

# class LSTMEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
#         super(LSTMEncoder, self).__init__()
#         self.lstm_layers = nn.ModuleList([
#             nn.LSTM(input_dim, hidden_dim, batch_first=True) for _ in range(num_layers)
#         ])
#         self.layer_norms = nn.ModuleList([
#             nn.LayerNorm(hidden_dim) for _ in range(num_layers)
#         ])
#         self.dropout = nn.Dropout(dropout)
#         self.hidden_dim = hidden_dim
#
#     def forward(self, x, x_mark_enc):
#         batch_size, seq_len, _ = x.size()
#         assert seq_len == len(self.lstm_layers), f"Expected sequence length {len(self.lstm_layers)}, got {seq_len}"
#
#         # Initialize hidden states and cell states for each LSTM layer
#         h = [torch.zeros(1, batch_size, self.hidden_dim).to(x.device) for _ in range(len(self.lstm_layers))]
#         c = [torch.zeros(1, batch_size, self.hidden_dim).to(x.device) for _ in range(len(self.lstm_layers))]
#
#         outputs = []
#         for i in range(seq_len):
#             x_step = x[:, i, :].unsqueeze(1)  # [128, 1, 1024]
#             x_step, (h[i], c[i]) = self.lstm_layers[i](x_step, (h[i], c[i]))
#             x_step = self.layer_norms[i](x_step.squeeze(1))  # Remove the sequence dimension
#             x_step = self.dropout(x_step)
#             outputs.append(x_step)
#
#         outputs = torch.stack(outputs, dim=1)  # [128, 7, hidden_dim]
#         return outputs

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, x_mark_enc):
        output, (h_n, c_n) = self.lstm(x)
        output = output.transpose(1, 2)
        output = self.batch_norm(output)
        output = output.transpose(1, 2)
        output = self.linear(output)
        return output
