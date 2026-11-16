import math
import torch
import torch.nn.functional as F
import lab07_starter as S
from vae_lib import VAE, kl_divergence, vae_loss


def check_kl_monte_carlo(seed=0, n_samples=50000):
    torch.manual_seed(seed)
    mu = torch.randn(4, 8)
    logvar = torch.randn(4, 8) * 0.7
    closed = kl_divergence(mu, logvar)

    std = torch.exp(0.5 * logvar)
    eps = torch.randn(n_samples, 4, 8)
    z = mu.unsqueeze(0) + std.unsqueeze(0) * eps

    log_q = -0.5 * (math.log(2 * math.pi) + logvar.unsqueeze(0) + ((z - mu.unsqueeze(0)) / std.unsqueeze(0)) ** 2).sum(dim=-1)
    log_p = -0.5 * (math.log(2 * math.pi) + z ** 2).sum(dim=-1)
    mc = (log_q - log_p).mean(dim=0)

    max_diff = (closed - mc).abs().max().item()
    print("Check 1 - Closed-form KL vs Monte Carlo KL (50,000 samples):")
    print("  closed-form:", [round(v, 4) for v in closed.tolist()])
    print("  monte carlo:", [round(v, 4) for v in mc.tolist()])
    print("  max abs diff:", round(max_diff, 5))


def check_reparameterization_gradients(X_batch):
    torch.manual_seed(0)
    model_trick = VAE(d_in=64, d_hidden=64, d_latent=8)
    mu, logvar = model_trick.encode(X_batch)
    z_trick = model_trick.reparameterize(mu, logvar)
    recon_trick = F.binary_cross_entropy_with_logits(model_trick.decode(z_trick), X_batch, reduction="none").sum(dim=1).mean()
    model_trick.zero_grad()
    recon_trick.backward()
    grad_with_trick = model_trick.fc_mu.weight.grad.norm().item()

    torch.manual_seed(0)
    model_detached = VAE(d_in=64, d_hidden=64, d_latent=8)
    mu_d, logvar_d = model_detached.encode(X_batch)
    z_detached = model_detached.reparameterize(mu_d, logvar_d).detach()
    recon_detached = F.binary_cross_entropy_with_logits(model_detached.decode(z_detached), X_batch, reduction="none").sum(dim=1).mean()
    model_detached.zero_grad()
    recon_detached.backward()
    grad_detached = 0.0 if model_detached.fc_mu.weight.grad is None else model_detached.fc_mu.weight.grad.norm().item()

    print("Check 2 - Encoder fc_mu grad norm from reconstruction term:")
    print("  with reparameterization trick:", round(grad_with_trick, 6))
    print("  with detached z:              ", grad_detached)


def check_elbo_equivalence(X_batch):
    torch.manual_seed(42)
    model = VAE(d_in=64, d_hidden=64, d_latent=8)
    torch.manual_seed(123)
    loss_beta1 = vae_loss(model, X_batch, beta=1.0).item()

    torch.manual_seed(123)
    logits, mu, logvar = model(X_batch)
    recon = F.binary_cross_entropy_with_logits(logits, X_batch, reduction="none").sum(dim=1)
    kl = kl_divergence(mu, logvar)
    neg_elbo = (recon + kl).mean().item()

    print("Check 3 - Loss at beta=1 vs single-sample negative ELBO:")
    print("  vae_loss(beta=1):", round(loss_beta1, 6))
    print("  single-sample -ELBO:", round(neg_elbo, 6))
    print("  abs diff:", abs(loss_beta1 - neg_elbo))


if __name__ == "__main__":
    Xtr, _, _, _ = S.load_task()
    check_kl_monte_carlo()
    check_reparameterization_gradients(Xtr[:64])
    check_elbo_equivalence(Xtr[:64])