import torch
import torch.nn as nn
import math

class DiffusionModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Define beta schedule
        self.betas = self._cosine_beta_schedule(num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Noise prediction network
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1)
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def forward(self, x, t):
        return self.model(x)
    
    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,))
    
    def add_noise(self, x_start, t):
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alphas_cumprod[t])[:, None, None, None]
        noise = torch.randn_like(x_start)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise