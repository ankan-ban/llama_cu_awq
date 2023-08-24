# Import torch module
import torch

# Import os module
import os

# Load the file
data = torch.load("llama2-7b-awq.pt")

# Create a sub-directory named 'bin_wt' if it does not exist
os.makedirs('bin_wt', exist_ok=True)

# If the data is a dictionary, iterate over its keys and values
if isinstance(data, dict):
  for key, value in data.items():
    print(key, type(value))
    # If the value is a tensor, check its shape and dtype
    if isinstance(value, torch.Tensor):
      print(value.shape, value.dtype)
      # Dump the tensor to a binary file with the same name as the key
      with open(os.path.join('bin_wt', key + '.bin'), 'wb') as f:
        f.write(value.numpy().tobytes())
