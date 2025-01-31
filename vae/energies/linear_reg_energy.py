import torch
from jaxtyping import Float, Int, Bool
from torch import Tensor
import torch.distributions as D
from .linear_reg_utils import get_dataloaders, Sampler, FixedDataset
#todo implementar para dimensões maiores que 1 (x_dim; ie features > 1)

class LinearEnergy:
    def __init__(self, device, dim=2, set_size=10, batch_size=256, number_of_datasets=10):

        self.device = device
        self.data_ndim = dim
        self.batch_size = batch_size
        self.set_size = set_size
        self.likehood_var = 10
        self.x_prior = D.Uniform(-10, 10)
        self.theta_prior = D.Normal(0, 3)

        sampler = Sampler(number_of_datasets, self.x_prior, D.Normal(0, 3), self.likehood_var, dim - 1, set_size)
        train_dataloader, evaluation_subset = get_dataloaders(sampler, batch_size)

        self.train_dataloader = train_dataloader
        self.evaluation_dataloader = evaluation_subset


    def _extend_x(self, x):
        ones = torch.ones(x.shape[:-1] + (1,), device=x.device)
        return torch.cat([x, ones], dim=-1)

    def energy(self, state, condition):
        x = condition[:,:,:-1]
        y = condition[:,:,-1]
        x = self._extend_x(x)

        y_mean = (x @ state.unsqueeze(-1)).squeeze()
        log_likelihood = D.Normal(y_mean, self.likehood_var).log_prob(y).sum(-1)
        log_pior = self.theta_prior.log_prob(state).sum(-1)

        return -(log_likelihood + log_pior)

    def log_reward(self, state: Float[Tensor, "batch n_dim"], condition: Float[Tensor, "batch set_size n_dim"]):
        return -self.energy(state, condition)


    def sample_train_set(self):
        real_data, _ = next(iter(self.train_dataloader))
        return real_data

    def sample_evaluation_set(self):
        real_data, _ = next(iter(self.evaluation_dataloader))
        return real_data


    def sample(self, batch_size, evaluation=False):
        if evaluation:
            return self.sample_evaluation_set().to(self.device)
        return self.sample_train_set().to(self.device)


