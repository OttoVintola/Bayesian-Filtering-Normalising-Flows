import torch
from torch import nn
# from torch.nn import functional as F

class LorenzData():
    def __init__(self, dt=0.01, latent_dim=3, obs_dim=10, latent_noise=0.1, obs_noise=0.2, T=60):
        self.T = T
        self.dt = dt
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.latent_noise = latent_noise
        self.obs_noise = obs_noise

        self.emission_net = nn.Sequential(nn.Linear(latent_dim, obs_dim), nn.Softplus())
        self.skip_proj = nn.Linear(latent_dim, obs_dim)

        for p in self.emission_net.parameters(): p.requires_grad = False
        for p in self.skip_proj.parameters(): p.requires_grad = False
        
    def lorenz_step(self, z):
            # z: [1, 3]
            sigma, rho, beta = 10.0, 28.0, 8.0/3.0
            x, y, v = z[:, 0], z[:, 1], z[:, 2]
            dxdt = sigma * (y - x)
            dydt = x * (rho - v) - y
            dvdt = x * y - beta * v
            return torch.stack([dxdt, dydt, dvdt], dim=1)
    
    def generate(self, num_trials, T):
         
        obs = []
        latents = []
        for _ in range(num_trials):
            z = torch.zeros((self.T, self.latent_dim))

            # Make a random 
            z[0] = torch.randn(self.latent_dim) * 10
            

            for t in range(1, T):
                dz = self.lorenz_step(z[t-1:t]) * self.dt
                z[t] = dz * torch.randn(self.latent_dim) * self.latent_noise

            x = self.emission_net(z) + self.skip_proj(z)
            x += torch.randn_like(x) * self.obs_noise

            obs.append(x)
            latents.append(z)
         
        return torch.stack(obs), torch.stack(latents)

            
                    
                   
         
         

