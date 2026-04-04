import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal


# input dim = d
# f : R^2d -> R^2d
class PlanarFlow(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_dim, encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.pair_dim = 2 * latent_dim
        self.param_net = nn.Sequential(encoder, nn.Linear(hidden_dim, 2 * self.pair_dim + 1))

    def forward(self, z_pair, obs_pair):
        """
        z_pair is the concatenation of z_t and z_{t+1}
        obs_pair is the concatenation of x_t and x_{t+1}

        Returns:
        z_transformed: the transformed z_pair after applying the planar flow
        logdet: the log determinant of the Jacobian of the transformation, which is needed for
        """


        # phi
        params = self.param_net(obs_pair)
        u = params[:, :self.pair_dim]
        w = params[:, self.pair_dim : 2 * self.pair_dim]
        b = params[:, -1:]

    
        linear = b + torch.sum(w * z_pair, dim=1, keepdim=True)

        h = torch.tanh(linear)

        z_transformed = z_pair + u * h

        # Log determinant of the Jacobian
        h_prime = 1 - torch.tanh(linear) ** 2
        inner_prod = torch.sum(u * (h_prime * w), dim=1, keepdim=True)
        logdet = torch.log(torch.abs(1 + inner_prod))
        return z_transformed, logdet
        
class SharedEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.backbone = nn.Sequential(
            nn.Linear(2*obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if x.shape[-1] == self.obs_dim:
            x = torch.cat([x, x], dim=-1)
        return self.backbone(x)

class FilteringNormalizingFlow(nn.Module):
    def __init__(self, base_mean, base_var, batch_size, latent_dim, hidden_dim, obs_dim, encoder, T):
        super().__init__()
        self.transforms = nn.ModuleList([PlanarFlow(latent_dim, obs_dim, hidden_dim, encoder) for _ in range(T-1)])
        self.mean = base_mean
        self.var = base_var
        self.base_dist = MultivariateNormal(self.mean, self.var)
        self.T = T
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        # phi
        self.recognition_net = nn.Sequential(encoder, nn.Linear(hidden_dim, 2*self.latent_dim))
    
    def flow(self, z_pair, obs_pair):
        log_det_sum = 0
        z = z_pair
        for transform in self.transforms:
            z, logdet_layer = transform(z, obs_pair)
            log_det_sum += logdet_layer
        return z, log_det_sum
    

    def forward(self, observations):
        # sample_size = torch.Size([self.batch_size, self.T, self.latent_dim])
        # z = self.base_dist.sample(sample_size)

        # z is of shape (n_trials, T, 2*latent_dim)
        
        z = self.recognition_net(observations)
        mu, logvar = z[:, :, :self.latent_dim], z[:, :, self.latent_dim:]

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z_log_prob = -0.5 * (logvar + torch.pow(z - mu, 2) / torch.exp(logvar) + \
                                 np.log(2 * np.pi)).sum(dim=-1) # [batch, T]
        
        log_q = z_log_prob.sum(dim=1)
        
        T = observations.shape[1]

        logdet_sum = 0
        for t in range(T-1):
            z_pair = torch.cat([z[:, t, :], z[:, t+1, :]], dim=1)
            obs_pair = torch.cat([observations[:, t, :], observations[:, t+1, :]], dim=1)

            z_pair_transformed, logdet = self.flow(z_pair, obs_pair)

            z[:, t, :] = z_pair_transformed[:, :self.latent_dim]
            z[:, t+1, :] = z_pair_transformed[:, self.latent_dim:]

            logdet_sum += logdet

        return z, log_q - logdet_sum


class AEVB(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_dim, base_mean, base_var, batch_size, T):
        super().__init__()
        encoder = SharedEncoder(obs_dim, hidden_dim)

        self.flow_module = FilteringNormalizingFlow(base_mean, base_var, batch_size, latent_dim, hidden_dim, obs_dim, encoder, T)
        # self.recognition_net = nn.Sequential(encoder, nn.Linear(hidden_dim, latent_dim))
        self.transition_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # theta
        self.emission_net = nn.Sequential(
                    nn.Linear(latent_dim, obs_dim), 
                    nn.Softplus()
        )
        self.skip_proj = nn.Linear(latent_dim, obs_dim)
        self.optimizer: torch.optim.Optimizer | None = None  # Will be set later


    def log_pz(self, z, Q_sigma=0.1, Q_0_sigma=1.0):

        batch_size, T, latent_dim = z.shape

        log_p_z1 = -0.5 * torch.sum((z[:, 0, :] ** 2) / Q_0_sigma**2 + latent_dim * np.log(2 * np.pi * Q_0_sigma**2))

        z_prev = z[:, :-1, :]
        z_curr = z[:, 1:, :]
        mu_t = self.transition_net(z_prev) + z_prev

        term = torch.sum((z_curr - mu_t)**2 / Q_sigma**2, dim=(1, 2))
        const = (T - 1) * latent_dim * np.log(2 * np.pi * Q_sigma**2)
        log_p_transitions = -0.5 * (term + const)

        return log_p_z1 + log_p_transitions


    def elbo(self, observations, obs_noise_var=0.2**2):
        batch_size, T, obs_dim = observations.shape

        z_sampled, log_q_zx = self.flow_module(observations)

        x_reconstructed = self.emission_net(z_sampled) + self.skip_proj(z_sampled)
        log_p_xz = -0.5 * (torch.sum((observations - x_reconstructed)**2 / obs_noise_var, dim=(1, 2)) + 
                               T * obs_dim * np.log(2 * np.pi * obs_noise_var))

        log_p_z = self.log_pz(z_sampled)

        return (log_p_xz + log_p_z - log_q_zx).mean()
    
    def training_step(self, observations, obs_noise_var=0.2**2):
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been set on AEVB before training_step().")

        self.optimizer.zero_grad()
        loss = -self.elbo(observations, obs_noise_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()



