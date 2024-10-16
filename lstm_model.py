import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from pprint import pprint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

output_dim = 1  # binary classification for thumbs up or down
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for classification as a thumbs up

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "lstm_model_weights.json"


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def split_feature_label(data):
    # Assuming the last feature in the last time step is the label
    X = data[:, :, :-1]  # All features except the last one in each time step
    Y = data[:, -1, -1]  # The last feature of the last time step
    return X, Y


# Loader fn
def load_data(dataset, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def main():
    train_path = "train_data/train_sequences_0.pt"
    test_path = "test_data/test_sequences_0.pt"
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    batch_size = 64
    n_iters = len(train_data) * 5  # 5 epochs
    num_epochs = int(n_iters / (len(train_data) / batch_size))
    
    # Assuming the data shape is (num_sequences, sequence_length, num_features)
    # and the last feature in each sequence is the label
    X_train, y_train = split_feature_label(train_data)
    X_test, y_test = split_feature_label(test_data)
    
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), shuffle=True, batch_size=64
    )
    
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), shuffle=True, batch_size=64
    )
    
    lstm_model = LSTM_Model(input_dim, 100, output_dim)
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.0004
    optimizer = torch.optim.SGD(lstm_model.parameters(), lr=learning_rate)
    iter = 0
    
    for epoch in range(num_epochs):
        for i, (X, Y) in enumerate(train_loader):
            Y = Y.view(-1, 1)
            optimizer.zero_grad()
            outputs = lstm_model(X.float())
            loss = criterion(outputs, Y.float())
            loss.backward()
            optimizer.step()
            iter += 1

            if iter % 500 == 0:
                correct = 0
                total = 0
                all_labels = []
                all_probs = []
                for X, Y in test_loader:
                    outputs = lstm_model(X.float())
                    probs = outputs.detach().numpy().flatten()
                    predicted = (outputs > detect_threshold).float()
                    total += Y.size(0)
                    correct += (predicted == Y.view(-1, 1)).sum().item()
                    all_labels.extend(Y.numpy())
                    all_probs.extend(probs)

                accuracy = 100 * correct / total
                auc_roc = roc_auc_score(all_labels, all_probs)
                precision, recall, _ = precision_recall_curve(all_labels, all_probs)
                auc_pr = auc(recall, precision)
                print(
                    "Iteration: {}. Loss: {}. Accuracy: {}. AUC-ROC: {:.4f}. AUC-PR: {:.4f}".format(
                        iter, loss.item(), accuracy, auc_roc, auc_pr
                    )
                )
                
    # Extract the model's state dictionary, convert to JSON serializable format
    state_dict = model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

    # Store state dictionary
    with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
        json.dump(serializable_state_dict, f)

    print("\n--- Model Training Complete ---")
    print("\nModel weights saved to ", SAVE_MODEL_PATH + SAVE_MODEL_FILENAME)


if __name__ == "__main__":
    main()