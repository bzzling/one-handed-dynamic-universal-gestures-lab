import torch

sequences_tensor = torch.load("test_data/test_sequences_0.pt")
tensor = torch.load("test_data/test_0.pt")

sequence_shape = sequences_tensor.shape
shape = tensor.shape

print(f"Tensor Sequences Shape: {sequence_shape}")
print(f"Tensor Shape: {shape}")