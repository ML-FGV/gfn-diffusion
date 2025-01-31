import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.distributions as D
import matplotlib.pyplot as plt

class FixedDataset(Dataset):
    def __init__(self, sample_prior_fn, seed):
        self.sample_prior = sample_prior_fn
        self.seed = seed
        self.data, self.theta = self._generate_data()

    def _generate_data(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        data, theta = self.sample_prior()

        return data, theta

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.theta[idx]


class Sampler:
    def __init__(self, n_datasets, x_prior, theta_prior, likelihood_var, x_shape, set_size):
        self.n_datasets = n_datasets
        self.x_prior = x_prior
        self.theta_prior = theta_prior
        self.likelihood_var = likelihood_var
        self.x_shape = x_shape
        self.set_size = set_size

    def sample(self):
        x = self.x_prior.sample((self.n_datasets, self.set_size, self.x_shape))
        x = _extend_x(x)
        theta = self.theta_prior.sample((self.n_datasets, self.x_shape + 1))
        y_mean = (x @ theta.unsqueeze(-1)).squeeze()
        y = torch.distributions.Normal(y_mean, self.likelihood_var).sample()
        data_set = torch.cat([x[:,:,:-1], y.unsqueeze(-1)], dim=-1)
        return data_set, theta

def _extend_x(x):
    ones = torch.ones(x.shape[:-1] + (1,), device=x.device)
    return torch.cat([x, ones], dim=-1)

def get_dataloaders(sampler, batch_size, seed1=25, seed2=42):

    train_dataset = FixedDataset(sampler.sample, seed=seed1)
    train_random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=10000)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_random_sampler)

    eval_dataset = FixedDataset(sampler.sample, seed=seed2)
    eval_random_sampler = RandomSampler(eval_dataset, replacement=True, num_samples=10000)
    eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_random_sampler)

    return train_data_loader, eval_data_loader


if __name__ == "__main__":

    x_prior = D.Normal(0, 1)
    theta_prior = D.Normal(0, 1)
    set_size = 10
    x_shape = 1
    likelihood_var = 10
    n_datasets = 10
    seed = 1

    sampler = Sampler(n_datasets, x_prior, theta_prior, likelihood_var, x_shape, set_size)
    dataset = FixedDataset(sampler.sample, seed=seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=200)
    data_loader = DataLoader(dataset, batch_size=20, sampler=sampler)

    fig, ax = plt.subplots(2, 5, figsize=(21, 10), sharey=True)
    ax = ax.flatten()
    for i, (data, theta) in enumerate(data_loader):
        ax[i].scatter(data[0,:,:-1], data[0,:,-1])
        ax[i].set_title(f"Dataset {i}")

    plt.show()

    fig, ax = plt.subplots(2, 5, figsize=(21, 10), sharey=True)
    ax = ax.flatten()
    for i, (data, theta) in enumerate(data_loader):
        ax[i].scatter(data[0, :, :-1], data[0, :, -1])
        ax[i].set_title(f"Dataset {i}")

    plt.show()
