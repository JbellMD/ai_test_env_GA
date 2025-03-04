import torch
from models.diffusion_models.basic_diffusion import DiffusionModel

def test_diffusion_model():
    in_channels = 1
    hidden_dim = 64
    batch_size = 16
    img_size = 28
    
    model = DiffusionModel(in_channels, hidden_dim)
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    t = torch.randint(0, model.num_timesteps, (batch_size,))
    output = model(x, t)
    
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"