"""VAE and conditional-VAE architectures for Fashion-MNIST.

Both share a small encoder/decoder of conv + dense layers, parameterise
a 32-dim latent gaussian, and use the β-VAE objective. The conditional
variant concatenates a 10-dim one-hot class onto the latent vector
before decoding, which lets us sample from a *targeted* class at
generation time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, cond_dim: int = 0) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7), nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # logits
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim, cond_dim=0)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        logits = self.decoder(z)
        return torch.sigmoid(logits)


class CVAE(nn.Module):
    """Conditional VAE: condition the decoder on a 10-dim class one-hot."""

    def __init__(self, latent_dim: int = 32, n_classes: int = 10) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim, cond_dim=n_classes)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        oh = F.one_hot(y, self.n_classes).float()
        return self.decoder(torch.cat([z, oh], dim=1)), mu, logvar

    @torch.no_grad()
    def sample_class(self, n: int, label: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        oh = F.one_hot(torch.full((n,), label, device=device), self.n_classes).float()
        logits = self.decoder(torch.cat([z, oh], dim=1))
        return torch.sigmoid(logits)


def elbo_loss(x_recon_logits: torch.Tensor, x: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """β-VAE loss = recon_BCE + β * KL(q(z|x) || N(0, I)).  Mean per batch."""
    recon = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon, kl
