import torch
import torch.nn as nn
from torch.nn import functional as F

class DSN(nn.Module):
    """Deep Summarization Network (DSN) - Architecture officielle"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell doit être 'lstm' ou 'gru'"
        
        # Le BiLSTM ou BiGRU
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
            
        # Couche linéaire pour transformer la sortie du RNN en un score d'importance unique
        # hid_dim * 2 car le réseau est bidirectionnel
        self.fc = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, in_dim)
        h, _ = self.rnn(x) # h shape: (batch, seq_len, hid_dim * 2)
        
        # Passage dans la couche linéaire puis transformation en probabilité (0 à 1)
        p = torch.sigmoid(self.fc(h))
        return p
