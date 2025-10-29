class RNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10, bidirectional=False, batch_first=True):
        super(RNN, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size,
                           batch_first=True, bidirectional=bidirectional)
        
        hidden_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28) -> (batch, 28, 28)
        x = x.squeeze(1)

        # RNN expects (batch, seq_len, input_size)
        out, _ = self.rnn(x)

        # Take the last time step
        if self.bidirectional:
                # split forward and backward
                forward_out = out[:, :, :self.hidden_size]   # (B, W, hidden_size)
                backward_out = out[:, :, self.hidden_size:]  # (B, W, hidden_size)
                # forward last
                forward_last = forward_out[:, -1, :]        # (B, hidden_size)
                # backward last (which is backward after reading first token in forward sequence)
                # backward “last” effectively means backward_out[:, 0, :]
                backward_last = backward_out[:, 0, :]        # (B, hidden_size)
                out = torch.cat((forward_last, backward_last), dim=-1) # # (B, hidden_size * directions)
        else:
             out = out[:, -1, :]

        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out