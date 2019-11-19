import torch
import torch.utils.data
from torch.distributions import Normal

def add_noise_to_dataloader(dataloader, noise_deviation=0.1, seed=0):
    torch.manual_seed(seed)

    X, Y = dataloader.dataset.tensors

    noise = Normal(0, noise_deviation).sample(X.shape)
    X_with_noise = X + noise

    tensor_dataset = torch.utils.data.TensorDataset(X_with_noise, Y)
    dataloader_with_noise = torch.utils.data.DataLoader(tensor_dataset, shuffle=False)

    return dataloader_with_noise
