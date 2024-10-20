import torch
import torch.nn as nn
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

print("libs loaded")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"device: {device}")

output_dim = 1  # binary classification for thumbs up or down
input_dim = 17  # 17 features
detect_threshold = 0.7  # threshold for classification as a thumbs up

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "lstm_model_weights.json"


def save_model(model, path, filename):
    # Extract the model's state dictionary, convert to JSON serializable format
    state_dict = model.state_dict()
    serializable_state_dict = {key: value.cpu().tolist() for key, value in state_dict.items()}

    # Store state dictionary
    with open(path + filename, "w") as f:
        json.dump(serializable_state_dict, f)

    print("\n--- Model Training Complete ---")
    print("\nModel weights saved to ", path + filename)

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(dim=0), self.hidden_dim).to(device) #initialized hidden state
        c0 = torch.zeros(self.num_layers, x.size(dim=0), self.hidden_dim).to(device) #initialized cell state
        
        output, states = self.lstm(x, (h0, c0)) # states represents hidden and cell states (not needed)
        output = self.fc(out[:, -1, :]) # get the last time step's output for each sequence
        return output


def split_feature_label(data):
    X = data[:, :, :-1]
    Y = data[:, -1, -1]
    print("features split")
    return X, Y


def main():
    print("starting...")
    train_path = "train_data/train_sequences_0.pt"
    test_path = "test_data/test_sequences_0.pt"
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    print("data found")
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

    print("data loaded")
    
    lstm_model = LSTM_Model(input_dim, 100, output_dim)
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = 0.0004
    optimizer = torch.optim.SGD(lstm_model.parameters(), lr=learning_rate)
    iter = 0

    lstm_model.to(device)

    print("about to enter training loop")

    for epoch in range(num_epochs):
        for i, (X, Y) in enumerate(train_loader):
            lstm_model.train()
            X, Y = X.to(device), Y.to(device)
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

                lstm_model.eval()
                with torch.inference_mode():
                    for X, Y in test_loader:
                        X, Y = X.to(device), Y.to(device)
                        outputs = lstm_model(X.float())
                        probs = outputs.detach().cpu().numpy().flatten()
                        predicted = (outputs > detect_threshold).float()
                        total += Y.size(0)
                        correct += (predicted == Y.view(-1, 1)).sum().item()
                        all_labels.extend(Y.cpu().numpy())
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
    # Save the model
    save_model(lstm_model, SAVE_MODEL_PATH, SAVE_MODEL_FILENAME)


if __name__ == "__main__":
    main()