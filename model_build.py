import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, 50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(50, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)


if __name__ == "__main__":
    from data_setup import load_data
    X_train, _, _, _ = load_data()
    input_dim = X_train.shape[2]
    model = LSTMModel(input_dim)
    print(model)
