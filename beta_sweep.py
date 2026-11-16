import torch
import lab07_starter as S
from vae_lib import VAE, vae_loss, evaluate_vae_terms

BETAS = (0.1, 0.5, 1.0, 2.0, 4.0)
SEEDS = (0, 1, 2)


def main():
    Xtr, ytr, Xte, _ = S.load_task(seed=0, test_fraction=0.2)
    net = S.train_feature_net(Xtr, ytr, epochs=60, seed=0)
    fte = S.get_features(net, Xte)

    for beta in BETAS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            vae = VAE(d_in=64, d_hidden=64, d_latent=8)
            S.train(vae, lambda m, xb, yb, b=beta: vae_loss(m, xb, yb, beta=b),
                    Xtr, epochs=200, lr=2e-3, batch_size=64, seed=seed)

            recon_te, kl_te, _ = evaluate_vae_terms(vae, Xte)
            _, _, active_dims = evaluate_vae_terms(vae, Xtr)
            samples = vae.sample(len(Xte))
            scores = S.score_samples(samples, fte, net, Xtr)

            print(f"beta={beta:<3} | seed={seed} | recon={recon_te:.3f} | kl={kl_te:.3f} | "
                  f"active={active_dims}/8 | frechet={scores['frechet']:.3f} | "
                  f"prec={scores['precision']:.3f} | rec={scores['recall']:.3f}")

            if beta == 4.0 and seed == 0:
                S.save_grid(samples[:30], "fig_collapsed_samples.png", ncols=10,
                            title=f"Collapsed VAE samples (beta={beta}, seed={seed})")


if __name__ == "__main__":
    main()