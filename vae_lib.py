import torch
import torch.nn as nn
import torch.nn.functional as F
import lab07_starter as S


class VAE(nn.Module):
    """Variational Autoencoder with 1 hidden layer (64 units) on encoder and decoder."""

    def __init__(self, d_in=64, d_hidden=64, d_latent=8):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.enc_net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(d_hidden, d_latent)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)
        self.dec_net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_in),
        )

    def encode(self, x):
        h = self.enc_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec_net(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    @torch.no_grad()
    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return torch.sigmoid(self.decode(mu))

    @torch.no_grad()
    def sample(self, n, device=None):
        z = torch.randn(n, self.d_latent, device=device)
        return torch.sigmoid(self.decode(z))


def kl_divergence(mu, logvar):
    """Closed-form KL(N(mu, sigma^2) || N(0, I)) summed over latent dims, shape (B,)."""
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def vae_loss(model, xb, yb=None, beta=1.0):
    """(Reconstruction NLL summed over pixels + beta * KL summed over latent dims).mean()."""
    logits, mu, logvar = model(xb)
    recon = F.binary_cross_entropy_with_logits(logits, xb, reduction="none").sum(dim=1)
    kl = kl_divergence(mu, logvar)
    return (recon + beta * kl).mean()


@torch.no_grad()
def evaluate_vae_terms(model, X):
    """Return (mean_recon, mean_kl, active_dims) on dataset X."""
    logits, mu, logvar = model(X)
    recon = F.binary_cross_entropy_with_logits(logits, X, reduction="none").sum(dim=1).mean().item()
    kl = kl_divergence(mu, logvar).mean().item()
    active_dims = int((mu.var(dim=0, unbiased=False) > 0.01).sum().item())
    return recon, kl, active_dims


class CVAE(nn.Module):
    """Conditional VAE receiving one-hot class labels in both encoder and decoder."""

    def __init__(self, d_in=64, d_hidden=64, d_latent=8, n_classes=10):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.n_classes = n_classes
        self.enc_net = nn.Sequential(
            nn.Linear(d_in + n_classes, d_hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(d_hidden, d_latent)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)
        self.dec_net = nn.Sequential(
            nn.Linear(d_latent + n_classes, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_in),
        )

    def encode(self, x, y_oh):
        h = self.enc_net(torch.cat([x, y_oh], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y_oh):
        return self.dec_net(torch.cat([z, y_oh], dim=1))

    def forward(self, x, y_oh):
        mu, logvar = self.encode(x, y_oh)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, y_oh)
        return logits, mu, logvar

    @torch.no_grad()
    def sample_class(self, digit, n, device=None):
        z = torch.randn(n, self.d_latent, device=device)
        y = torch.full((n,), digit, dtype=torch.long, device=device)
        y_oh = S.one_hot(y, self.n_classes)
        return torch.sigmoid(self.decode(z, y_oh))


def cvae_loss(model, xb, yb, beta=1.0):
    """Conditional VAE loss with one-hot labels."""
    y_oh = S.one_hot(yb, model.n_classes)
    logits, mu, logvar = model(xb, y_oh)
    recon = F.binary_cross_entropy_with_logits(logits, xb, reduction="none").sum(dim=1)
    kl = kl_divergence(mu, logvar)
    return (recon + beta * kl).mean()