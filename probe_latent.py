import torch
import lab07_starter as S
from vae_lib import VAE, vae_loss, CVAE, cvae_loss

SEEDS = (0, 1, 2)


def get_different_digit_pairs(Xte, yte, n_pairs=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    pairs = []
    idx = torch.randperm(len(Xte), generator=g).tolist()
    used = set()
    for i in idx:
        if i in used:
            continue
        for j in idx:
            if j != i and j not in used and yte[i].item() != yte[j].item():
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
        if len(pairs) == n_pairs:
            break
    return pairs


@torch.no_grad()
def measure_interpolation(pairs, Xte, ae, vae, net, n_steps=10):
    ts = torch.linspace(0.0, 1.0, n_steps).unsqueeze(1)
    results = {}
    for name in ("pixel", "ae", "vae"):
        all_confs, int_confs, changes = [], [], 0
        for i, j in pairs:
            xa, xb = Xte[i:i + 1], Xte[j:j + 1]
            if name == "pixel":
                path = (1.0 - ts) * xa + ts * xb
            elif name == "ae":
                za, zb = ae.encode(xa), ae.encode(xb)
                path = ae.decode((1.0 - ts) * za + ts * zb).clamp(0, 1)
            else:
                za, _ = vae.encode(xa)
                zb, _ = vae.encode(xb)
                path = torch.sigmoid(vae.decode((1.0 - ts) * za + ts * zb))

            probs = torch.softmax(net(path), dim=1)
            confs, preds = probs.max(dim=1)
            all_confs.append(confs.mean().item())
            int_confs.append(confs[1:-1].mean().item())
            changes += int((preds[1:] != preds[:-1]).sum().item())
        results[name] = (sum(all_confs) / len(all_confs), sum(int_confs) / len(int_confs), changes)
    return results


def main():
    Xtr, ytr, Xte, yte = S.load_task(seed=0, test_fraction=0.2)
    net = S.train_feature_net(Xtr, ytr, epochs=60, seed=0)
    pairs = get_different_digit_pairs(Xte, yte, n_pairs=40, seed=0)

    print("=== Part A: 10-step Interpolation (40 different-digit pairs) ===")
    for seed in SEEDS:
        torch.manual_seed(seed)
        ae = S.AE(d_in=64, d_hidden=64, d_latent=8, linear=False)
        S.train(ae, S.ae_loss, Xtr, epochs=200, lr=2e-3, batch_size=64, seed=seed)

        torch.manual_seed(seed)
        vae = VAE(d_in=64, d_hidden=64, d_latent=8)
        S.train(vae, lambda m, xb, yb: vae_loss(m, xb, yb, beta=1.0), Xtr, epochs=200, lr=2e-3, batch_size=64, seed=seed)

        res = measure_interpolation(pairs, Xte, ae, vae, net, n_steps=10)
        for mname, (c_all, c_int, chg) in res.items():
            print(f"seed={seed} | {mname:<5} | conf_all={c_all:.4f} | conf_interior={c_int:.4f} | class_changes={chg}")

        if seed == 0:
            ts = torch.linspace(0.0, 1.0, 10).unsqueeze(1)
            i, j = pairs[0]
            xa, xb = Xte[i:i + 1], Xte[j:j + 1]
            za_ae, zb_ae = ae.encode(xa), ae.encode(xb)
            za_v, _ = vae.encode(xa)
            zb_v, _ = vae.encode(xb)
            p_pix = (1.0 - ts) * xa + ts * xb
            p_ae = ae.decode((1.0 - ts) * za_ae + ts * zb_ae).clamp(0, 1)
            p_vae = torch.sigmoid(vae.decode((1.0 - ts) * za_v + ts * zb_v))
            S.save_grid(torch.cat([p_pix, p_ae, p_vae], dim=0), "fig_interpolation.png", ncols=10,
                        title="Rows: Pixel cross-fade, AE latent, VAE latent (10 steps)")

    print("\n=== Part B: Conditional VAE vs Unconditional VAE ===")
    for seed in SEEDS:
        torch.manual_seed(seed)
        cvae = CVAE(d_in=64, d_hidden=64, d_latent=8, n_classes=10)
        S.train(cvae, lambda m, xb, yb: cvae_loss(m, xb, yb, beta=1.0), Xtr, y=ytr,
                epochs=200, lr=2e-3, batch_size=64, seed=seed)

        per_digit_acc = []
        grid_imgs = []
        with torch.no_grad():
            for d in range(10):
                imgs = cvae.sample_class(d, 200)
                acc = (net(imgs).argmax(dim=1) == d).float().mean().item()
                per_digit_acc.append(acc)
                if seed == 0:
                    grid_imgs.append(imgs[:10])
        mean_acc = sum(per_digit_acc) / 10.0
        print(f"seed={seed} | CVAE mean target share={mean_acc:.4f} | per-digit={[round(a, 3) for a in per_digit_acc]}")

        if seed == 0:
            S.save_grid(torch.cat(grid_imgs, dim=0), "fig_conditional_samples.png", ncols=10,
                        title="Conditional VAE prior samples (rows 0-9)")
            torch.manual_seed(0)
            vae = VAE(d_in=64, d_hidden=64, d_latent=8)
            S.train(vae, lambda m, xb, yb: vae_loss(m, xb, yb, beta=1.0), Xtr, epochs=200, lr=2e-3, batch_size=64, seed=0)
            with torch.no_grad():
                uncond_preds = net(vae.sample(1000)).argmax(dim=1)
                counts = [int((uncond_preds == d).sum().item()) for d in range(10)]
            print("Unconditional VAE class counts (1,000 samples, seed 0):", counts)


if __name__ == "__main__":
    main()