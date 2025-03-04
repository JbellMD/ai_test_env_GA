import torch
from models.gans.basic_gan import Generator, Discriminator

def test_generator():
    latent_dim = 100
    img_shape = (1, 28, 28)
    batch_size = 16
    
    generator = Generator(latent_dim, img_shape)
    z = torch.randn(batch_size, latent_dim)
    output = generator(z)
    
    assert output.shape == (batch_size, *img_shape), \
        f"Expected shape {(batch_size, *img_shape)}, got {output.shape}"

def test_discriminator():
    img_shape = (1, 28, 28)
    batch_size = 16
    
    discriminator = Discriminator(img_shape)
    img = torch.randn(batch_size, *img_shape)
    output = discriminator(img)
    
    assert output.shape == (batch_size, 1), \
        f"Expected shape {(batch_size, 1)}, got {output.shape}"