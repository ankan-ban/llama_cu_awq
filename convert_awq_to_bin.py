# Import torch module
import torch

# Import os module
import os

# Import sys module
import sys

# Get the filename from the command line argument
filename = sys.argv[1]

# Get the directory name from the command line argument
dirname = sys.argv[2]

# Load the data from the filename
data = torch.load(filename)

# Create a sub-directory with the given name if it does not exist
os.makedirs(dirname, exist_ok=True)

# If the data is a dictionary, iterate over its keys and values
if isinstance(data, dict):
  for key, value in data.items():
    print(key, type(value))
    # If the value is a tensor, check its shape and dtype
    if isinstance(value, torch.Tensor):
      print(value.shape, value.dtype)
      # Dump the tensor to a binary file with the same name as the key in the given directory
      with open(os.path.join(dirname, key + '.bin'), 'wb') as f:
        f.write(value.cpu().numpy().tobytes())
