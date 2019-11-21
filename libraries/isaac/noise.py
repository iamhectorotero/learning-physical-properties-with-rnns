import torch
import torch.utils.data
from torch.distributions import Normal

def add_noise_to_dataloader(dataloader, noise_deviation=0.1, seed=0):
    """Adds Gaussian noise with mean 0 to the dataloader.
    Args:
        dataloader: a torch DataLoader.
        noise_deviation: (default=0.1) The standard deviation of the Gaussian distribution.
        seed: the value that will be set to the torch rng before generating the noise values.
    Returns:
        dataloader_with_noise"""

    torch.manual_seed(seed)

    X, Y = dataloader.dataset.tensors

    noise = Normal(0, noise_deviation).sample(X.shape)
    if X.is_cuda:
        noise = noise.cuda()

    X_with_noise = X + noise

    tensor_dataset = torch.utils.data.TensorDataset(X_with_noise, Y)
    dataloader_with_noise = torch.utils.data.DataLoader(tensor_dataset, shuffle=False,
                                                        batch_size=dataloader.batch_size)

    return dataloader_with_noise
