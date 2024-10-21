import torch

FROM = "test_data/test_0.pt"
DESTINATION = "test_data/test_sequences_0.pt"

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return torch.stack(sequences)

if __name__ == "__main__":
    data = torch.load(f'{FROM}')
    sequences = create_sequences(data, seq_length=10)
    torch.save(sequences, f'{DESTINATION}')