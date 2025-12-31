import torch
import torch.nn as nn

class DSN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(DSN, self).__init__()
        # Le BiLSTM : analyse la structure temporelle
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Couche de sortie : transforme les 512 caractéristiques du LSTM (256x2 directions)
        # en une probabilité unique (0 à 1) via Sigmoid
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (1, T, 1024) -> T étant le nombre de frames
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x) # out: (1, T, 512)
        
        # On réduit à une probabilité par frame
        probs = self.sigmoid(self.fc(out)) # probs: (1, T, 1)
        return probs
