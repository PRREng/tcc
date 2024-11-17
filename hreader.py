import h5py
import numpy as np
import torch


def getData():
    # Open the .h5 file
    with h5py.File("data2.h5", "r") as f:
        # Access datasets
        x_dataset = f["x_data"]
        y_dataset = f["y_data"]

        # Convert datasets to NumPy arrays
        x_np = np.array(x_dataset[:])
        y_np = np.array(y_dataset[:])

        # Optional: Convert NumPy arrays to PyTorch tensors
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.long)

        x_tensor = x_tensor.permute(0, 2, 1).contiguous()
    return x_tensor, y_tensor
