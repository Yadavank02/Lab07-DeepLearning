import torch
import lab07_starter as S
from vae_lib import VAE, vae_loss

SEEDS = (0, 1, 2)


def main():
    Xtr, ytr, Xte, _ = S.load_task(seed=0, test_fraction=0.2)
    net = S.train_feature_net(Xtr, ytr, epochs=60, seed=0)
    fte = S.get_features(net, Xte)

    for seed in SEEDS:
        torch.manual_seed(seed)
        ae = S.AE(d_in=64, d_hidden=64, d_latent=8, linear=False)
        S.train(ae, S.ae_loss, Xtr, epochs=200, lr=2e-3, batch_size=64, seed=seed)
        ae_mse = S.reconstruction_mse(ae, Xte)
        with torch.no_grad():
            ae_samples = ae.decode(torch.randn(len(Xte), 8)).clamp(0, 1)
            ae_codes_tr = ae.encode(Xtr)
            ae_codes_te = ae.encode(Xte)
        ae_scores = S.score_samples(ae_samples, fte, net, Xtr)

        torch.manual_seed(seed)
        vae = VAE(d_in=64, d_hidden=64, d_latent=8)
        S.train(vae, lambda m, xb, yb: vae_loss(m, xb, yb, beta=1.0), Xtr, epochs=200, lr=2e-3, batch_size=64, seed=seed)
        vae_mse = ((vae.reconstruct(Xte) - Xte) ** 2).mean().item()
        with torch.no_grad():
            vae_samples = vae.sample(len(Xte))
            vae_mu_tr, _ = vae.encode(Xtr)
            vae_mu_te, _ = vae.encode(Xte)
        vae_scores = S.score_samples(vae_samples, fte, net, Xtr)

        print(f"=== Seed {seed} ===")
        print(f"AE  | test_mse={ae_mse:.4f} | " + " | ".join(f"{k}={v:.4f}" for k, v in ae_scores.items()))
        print(f"VAE | test_mse={vae_mse:.4f} | " + " | ".join(f"{k}={v:.4f}" for k, v in vae_scores.items()))
        print("  AE  train code std:", [round(x, 3) for x in ae_codes_tr.std(dim=0).tolist()])
        print("  AE  test  code std:", [round(x, 3) for x in ae_codes_te.std(dim=0).tolist()])
        print("  VAE train mu   std:", [round(x, 3) for x in vae_mu_tr.std(dim=0).tolist()])
        print("  VAE test  mu   std:", [round(x, 3) for x in vae_mu_te.std(dim=0).tolist()])

        if seed == 0:
            S.save_grid(ae_samples[:30], "fig_ae_samples.png", ncols=10, title="Plain AE prior samples (seed 0)")
            S.save_grid(vae_samples[:30], "fig_ae_vs_vae_samples.png", ncols=10, title="VAE (beta=1) prior samples (seed 0)")


if __name__ == "__main__":
    main()