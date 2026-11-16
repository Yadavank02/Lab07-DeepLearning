import numpy as np
import torch
import lab07_starter as S
from vae_lib import VAE, vae_loss


def main():
    Xtr, ytr, Xte, _ = S.load_task(seed=0, test_fraction=0.2)
    net = S.train_feature_net(Xtr, ytr, epochs=60, seed=0)
    fte = S.get_features(net, Xte)
    n_eval = len(Xte)  # 359

    torch.manual_seed(0)
    ae = S.AE(d_in=64, d_hidden=64, d_latent=8, linear=False)
    S.train(ae, S.ae_loss, Xtr, epochs=200, lr=2e-3, batch_size=64, seed=0)

    torch.manual_seed(0)
    vae_b1 = VAE(d_in=64, d_hidden=64, d_latent=8)
    S.train(vae_b1, lambda m, xb, yb: vae_loss(m, xb, yb, beta=1.0), Xtr, epochs=200, lr=2e-3, batch_size=64, seed=0)

    torch.manual_seed(0)
    vae_b4 = VAE(d_in=64, d_hidden=64, d_latent=8)
    S.train(vae_b4, lambda m, xb, yb: vae_loss(m, xb, yb, beta=4.0), Xtr, epochs=200, lr=2e-3, batch_size=64, seed=0)

    torch.manual_seed(0)
    real_sub = Xtr[:n_eval]
    copier = (real_sub + 0.02 * torch.randn_like(real_sub)).clamp(0, 1)
    mean_img = Xtr.mean(dim=0, keepdim=True).repeat(n_eval, 1)
    mean_noise = (mean_img + 0.1 * torch.randn_like(mean_img)).clamp(0, 1)

    with torch.no_grad():
        sets = {
            "1. Real train (359)": real_sub,
            "2. Copier (train + 0.02)": copier,
            "3. Mean + noise (0.1)": mean_noise,
            "4. Collapsed VAE (beta=4)": vae_b4.sample(n_eval),
            "5. VAE (beta=1)": vae_b1.sample(n_eval),
            "6. AE prior samples": ae.decode(torch.randn(n_eval, 8)).clamp(0, 1),
        }

    print("=== Part A: Scoring 6 sets of 359 images ===")
    for label, imgs in sets.items():
        sc = S.score_samples(imgs, fte, net, Xtr)
        print(f"{label:<28} | frechet={sc['frechet']:.3f} | prec={sc['precision']:.3f} | "
              f"rec={sc['recall']:.3f} | nearest_train={sc['nearest_train']:.3f}")

    print("\n=== Part B: Frechet distance vs subset size (10 random subsets each) ===")
    rng = np.random.RandomState(0)
    for size in (30, 60, 120, 359, 1000):
        fds = []
        for _ in range(10):
            idx = rng.choice(len(Xtr), size=size, replace=False)
            f_sub = S.get_features(net, Xtr[idx])
            fds.append(S.frechet_distance(fte, f_sub))
        print(f"N={size:<4} | mean_frechet={np.mean(fds):.3f} | std={np.std(fds):.3f}")


if __name__ == "__main__":
    main()