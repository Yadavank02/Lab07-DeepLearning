# Latent Spaces

A variational autoencoder (VAE) written from scratch on 8×8 digits (`sklearn.datasets.load_digits`), compared against a plain autoencoder (`AE`), swept across KL weights ($\beta \in \{0.1, 0.5, 1.0, 2.0, 4.0\}$), probed via latent-space interpolation and class conditioning, and evaluated with Fréchet distance, precision, recall, and nearest-training-image distance.

## Setup

### Using an existing or clean Python environment

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

## Run

Run `lab07_starter.py` once unmodified to verify the environment, then run the five experiment scripts in order:

```
python lab07_starter.py         # environment sanity check
python vae_checks.py            # piece 1: KL, gradient, and ELBO checks
python ae_vs_vae.py             # piece 2: same bottleneck (AE vs VAE)
python beta_sweep.py            # piece 3: KL weight sweep (beta = 0.1..4)
python probe_latent.py          # piece 4: interpolation and conditioning
python metrics_study.py         # piece 5: what the metrics capture

```

## Hardware

* **Device:** Apple MacBook Pro CPU (`macOS`)

* **Total runtime for full set of runs:** **1 minute 19 seconds (`79s`)** across all six scripts (`21:33:26` to `21:34:45 EST`)

## Conventions

* **Reconstruction term:** Binary cross-entropy with logits (`binary_cross_entropy_with_logits(..., reduction="none")`), **summed over all 64 pixels** per image, then **averaged over the batch**.

* **KL divergence term:** Closed-form KL to $\mathcal{N}(0, I)$, **summed over all 8 latent dimensions** per image, then **averaged over the batch**.

* **Total VAE loss:** $(\text{recon} + \beta \cdot \text{KL}).\text{mean}()$ over the batch.

* **Shared evaluation setup:** Every Fréchet distance, precision, recall, and nearest-training-image distance comparison uses the **same `FeatureNet`** (`train_feature_net(Xtr, ytr, epochs=60, seed=0)`) and the **same sample count (`n = 359`)** matching the test split (`load_task(seed=0, test_fraction=0.2)`), unless explicitly testing sample-size sensitivity in Piece 5.

* **Training budget:** `d_latent=8`, `d_hidden=64`, `epochs=200`, `lr=2e-3`, `batch_size=64`, `seeds=(0, 1, 2)`.

## Prediction (written before running the sweep)

* **Where collapse is expected:** Because the KL term is summed over all 8 latent dimensions (not averaged), $\beta$ scales the full 8-dimensional KL penalty directly against the 64-pixel Bernoulli NLL. At $\beta = 0.1$ and $\beta = 0.5$, reconstruction dominates and all or most latent dimensions remain active. Around $\beta = 2.0$ to $\beta = 4.0$, the KL penalty becomes strong enough relative to the 64-pixel reconstruction gain that the encoder can minimize total loss by pushing posterior means $\mu \to 0$ and log-variances $\log \sigma^2 \to 0$ ($q(z\vert{}x) \approx \mathcal{N}(0, I)$), causing **posterior collapse** (active dimensions dropping toward 0 and decoded samples converging toward a blurry average digit).

* **Which** $\beta$ **is expected to give the best samples:** $\beta = 0.5$ or $\beta = 1.0$. At $\beta = 0.1$, the latent space may have gaps where $\mathcal{N}(0, I)$ prior samples land in unpopulated regions, hurting Fréchet distance and precision. At $\beta = 1.0$ (the true ELBO) or $\beta = 0.5$, the aggregate posterior stays close enough to $\mathcal{N}(0, I)$ for clean prior sampling while preserving enough active dimensions for sharp, diverse digits.

## Results

### Starter Sanity Check (`lab07_starter.py`)

* **Split:** `1,438` train images, `359` test images (`64` pixels each)

* **Baseline AE (100 epochs) test reconstruction MSE:** `0.0163`

* **`FeatureNet` test classification accuracy:** `0.972`

* **AE prior samples from** $\mathcal{N}(0, I)$**:** Fréchet = `86.599`, Precision = `0.412`, Recall = `0.78`, Nearest Train = `2.031`

### Piece 1: VAE Checks (`vae_checks.py`)

| Check | Measured Value | Expected / Pass Condition | Status | 
| ----- | ----- | ----- | ----- | 
| Closed-form KL vs. Monte Carlo KL (50k samples) | `[4.4979, 4.1409, 6.4186, 8.8143]` vs. `[4.5152, 4.1587, 6.4217, 8.7636]` (max diff: `0.05074`) | Agreement within MC sampling noise | Pass | 
| Encoder `fc_mu` grad norm (with reparameterization) | `0.915171` | Strictly `> 0.0` | Pass | 
| Encoder `fc_mu` grad norm (detached `z`, recon only) | `0.0` | Exactly `0.0` | Pass | 
| Loss at $\beta=1$ vs. single-sample `-ELBO` | `45.032066` vs. `45.032066` (abs diff: `0.0`) | Exact match (`< 1e-5`) | Pass | 

### Piece 2: Same Bottleneck — AE vs. VAE at $\beta = 1$ (`ae_vs_vae.py`)

| Model | Seed | Test Recon MSE | Fréchet ($\downarrow$) | Precision ($\uparrow$) | Recall ($\uparrow$) | Nearest Train ($\uparrow$) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| Plain AE | 0 | 0.0153 | 74.4909 | 0.4262 | 0.7911 | 2.0409 | 
| Plain AE | 1 | 0.0151 | 98.9821 | 0.3621 | 0.7298 | 2.1781 | 
| Plain AE | 2 | 0.0149 | 108.8906 | 0.4262 | 0.8106 | 2.3455 | 
| VAE ($\beta=1$) | 0 | 0.0259 | 11.6216 | 0.8969 | 0.4345 | 1.1370 | 
| VAE ($\beta=1$) | 1 | 0.0265 | 11.9757 | 0.8858 | 0.4930 | 1.1212 | 
| VAE ($\beta=1$) | 2 | 0.0263 | 13.4196 | 0.8914 | 0.5070 | 1.1065 | 

**Code spread (per-dimension std of latent codes across `Xtr` and `Xte`):**

* **Seed 0 — AE train / test std:** `[0.685, 0.875, 0.574, 0.642, 0.687, 0.822, 0.532, 0.679]` / `[0.658, 0.857, 0.578, 0.693, 0.682, 0.789, 0.524, 0.689]`

* **Seed 0 — VAE train / test** $\mu$ **std:** `[0.94, 0.963, 0.936, 0.481, 0.021, 0.821, 0.02, 0.034]` / `[0.911, 0.953, 0.901, 0.496, 0.022, 0.855, 0.02, 0.033]`

* **Seed 1 — AE train / test std:** `[0.631, 0.574, 0.755, 0.645, 1.088, 0.658, 0.745, 0.646]` / `[0.629, 0.609, 0.764, 0.652, 1.112, 0.648, 0.686, 0.699]`

* **Seed 1 — VAE train / test** $\mu$ **std:** `[0.904, 0.445, 0.812, 0.912, 0.894, 0.021, 0.025, 0.038]` / `[0.895, 0.446, 0.844, 0.871, 0.88, 0.021, 0.025, 0.037]`

* **Seed 2 — AE train / test std:** `[0.719, 0.763, 0.753, 0.706, 0.61, 0.707, 0.767, 0.73]` / `[0.752, 0.715, 0.752, 0.672, 0.614, 0.723, 0.757, 0.743]`

* **Seed 2 — VAE train / test** $\mu$ **std:** `[0.895, 0.912, 0.907, 0.026, 0.534, 0.817, 0.031, 0.022]` / `[0.931, 0.879, 0.842, 0.027, 0.536, 0.849, 0.03, 0.024]`

* **Why one generates and the other cannot:** The plain AE achieves lower test reconstruction MSE (`0.0149`–`0.0153` vs. `0.0259`–`0.0265` for VAE) because it is unregularized, spreading its codes across all 8 dimensions (`std = 0.53`–`1.09`) with unpopulated holes between digit clusters—so decoding $\mathcal{N}(0, I)$ produces off-manifold artifacts (`Fréchet = 74.49`–`108.89`, `Precision = 0.36`–`0.43`). The VAE regularizes 5 active dimensions toward $\mathcal{N}(0, I)$ (`std ≈ 0.45`–`0.96`) while pruning 3 redundant dimensions (`std ≈ 0.02`–`0.04`), cutting Fréchet distance to `11.62`–`13.42` and raising precision to `0.89`.

### Piece 3: KL Weight Sweep (`beta_sweep.py`)

| $\beta$ | Seed | Test Recon | Test KL | Active Dims ($\text{var}(\mu) > 0.01$) | Fréchet ($\downarrow$) | Precision ($\uparrow$) | Recall ($\uparrow$) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| 0.1 | 0 | 17.240 | 14.226 | 8/8 | 10.515 | 0.758 | 0.864 | 
| 0.1 | 1 | 17.109 | 14.195 | 8/8 | 14.548 | 0.719 | 0.883 | 
| 0.1 | 2 | 17.099 | 14.277 | 8/8 | 11.340 | 0.763 | 0.852 | 
| 0.5 | 0 | 19.342 | 5.833 | 7/8 | 9.803 | 0.841 | 0.755 | 
| 0.5 | 1 | 19.255 | 5.955 | 7/8 | 8.695 | 0.844 | 0.749 | 
| 0.5 | 2 | 19.111 | 6.141 | 7/8 | 7.977 | 0.850 | 0.699 | 
| 1.0 | 0 | 20.971 | 3.483 | 5/8 | 12.805 | 0.922 | 0.437 | 
| 1.0 | 1 | 21.292 | 3.221 | 5/8 | 13.870 | 0.883 | 0.476 | 
| 1.0 | 2 | 21.313 | 3.289 | 5/8 | 13.694 | 0.900 | 0.412 | 
| 2.0 | 0 | 24.212 | 1.269 | 3/8 | 42.881 | 0.939 | 0.106 | 
| 2.0 | 1 | 24.454 | 1.159 | 3/8 | 44.559 | 0.905 | 0.109 | 
| 2.0 | 2 | 24.209 | 1.257 | 4/8 | 45.167 | 0.928 | 0.164 | 
| 4.0 | 0 | 27.250 | 0.001 | 0/8 | 218.790 | 1.000 | 0.000 | 
| 4.0 | 1 | 27.259 | 0.001 | 0/8 | 219.342 | 1.000 | 0.000 | 
| 4.0 | 2 | 27.247 | 0.000 | 0/8 | 224.166 | 1.000 | 0.000 | 

* **Sweep finding:** Our pre-sweep prediction matched the observed data. $\beta = 0.5$ achieved the best Fréchet distance (`7.977`–`9.803`) with `7/8` active dimensions, strong precision (`0.841`–`0.850`), and high recall (`0.699`–`0.755`). Partial collapse began at $\beta = 2.0$ (`3–4/8` active dims, `Fréchet = 42.88`–`45.17`, `Recall = 0.11`–`0.16`), and complete posterior collapse occurred at $\beta = 4.0$ across all three seeds (`0/8` active dims, `KL ≈ 0.000`–`0.001`, `Fréchet = 218.79`–`224.17`, `Recall = 0.000`, `Precision = 1.000`).

### Piece 4: Probing the Latent Space (`probe_latent.py`)

**Part A — 10-step Interpolation (40 test pairs of different digits):**

| Method | Seed | Mean Confidence (All 10 Steps) | Mean Confidence (Interior 8 Steps) | Total Class Changes | 
| ----- | ----- | ----- | ----- | ----- | 
| Pixel Cross-Fade | 0 | 0.9364 | 0.9244 | 50 | 
| Pixel Cross-Fade | 1 | 0.9364 | 0.9244 | 50 | 
| Pixel Cross-Fade | 2 | 0.9364 | 0.9244 | 50 | 
| AE Latent Path | 0 | 0.9323 | 0.9195 | 66 | 
| AE Latent Path | 1 | 0.9388 | 0.9277 | 63 | 
| AE Latent Path | 2 | 0.9309 | 0.9216 | 68 | 
| VAE Latent Path | 0 | 0.9173 | 0.9097 | 64 | 
| VAE Latent Path | 1 | 0.9272 | 0.9225 | 62 | 
| VAE Latent Path | 2 | 0.9242 | 0.9144 | 61 | 

* **Interpolation finding:** Across all three seeds, the VAE latent path **did not beat** the pixel cross-fade baseline on `FeatureNet` confidence (`0.9173`–`0.9272` all-step / `0.9097`–`0.9225` interior vs. `0.9364` / `0.9244` for pixel cross-fade). However, both latent paths produced noticeably more class transitions across the 40 pairs (`61`–`64` for VAE and `63`–`68` for AE vs. `50` for pixel cross-fade), because moving through the latent space traverses intermediate digit regions rather than simply fading one digit's pixels out while fading the other in.

**Part B — Conditional VAE (200 prior samples per digit vs. 1,000 unconditional VAE samples):**

* **Conditional VAE target-digit accuracy (mean across digits 0–9):**

  * **Seed 0:** `0.9995` — per-digit `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.995, 1.0]`

  * **Seed 1:** `1.0000` — per-digit `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`

  * **Seed 2:** `0.9995` — per-digit `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.995]`

* **Unconditional VAE class counts across 1,000 prior samples (Seed 0, digits 0–9):**

  * `[95, 96, 96, 126, 94, 80, 117, 94, 113, 89]` (counts vary from `80` for digit `5` to `126` for digit `3`, with no way to request a specific digit).

### Piece 5: What the Metrics Capture (`metrics_study.py`)

**Six 359-image sets:**

| Sample Set (n=359) | Fréchet ($\downarrow$) | Precision ($\uparrow$) | Recall ($\uparrow$) | Nearest Train ($\uparrow$) | 
| ----- | ----- | ----- | ----- | ----- | 
| 1\. Real train (359) | 2.694 | 0.905 | 0.928 | 0.000 | 
| 2\. Copier (`train + N(0, 0.02^2)`) | 2.390 | 0.905 | 0.914 | 0.134 | 
| 3\. Mean + noise (`0.1`) | 139.826 | 0.922 | 0.000 | 1.609 | 
| 4\. Collapsed VAE ($\beta=4$) | 218.769 | 1.000 | 0.000 | 1.511 | 
| 5\. VAE ($\beta=1$) | 13.677 | 0.886 | 0.457 | 1.136 | 
| 6\. AE prior samples | 69.268 | 0.451 | 0.827 | 2.025 | 

**Fréchet distance vs. subset size (10 random subsets of `Xtr` against `Xte`):**

| Subset Size ($N$) | Mean Fréchet ($\pm$ std across 10 subsets) | 
| ----- | ----- | 
| 30 | `22.916 ± 8.477` | 
| 60 | `10.487 ± 3.393` | 
| 120 | `4.523 ± 0.827` | 
| 359 | `2.968 ± 0.398` | 
| 1000 | `1.671 ± 0.183` | 

**Analysis of metrics (Piece 5):**
Fréchet distance compares the mean and covariance of real and generated `FeatureNet` embeddings, catching distributional mismatch (collapsed VAE `218.769`, noisy mean `139.826`, AE prior `69.268`) while remaining sensitive to sample-size estimation bias (dropping from `22.916 ± 8.477` at $N=30$ to `1.671 ± 0.183` at $N=1000$). Precision checks whether generated samples land inside the real $k$-NN feature manifold (catching off-manifold AE prior samples at `0.451`), whereas recall checks coverage of real diversity (catching mode collapse where precision looks misleadingly high—collapsed VAE scores `1.000` precision and mean-plus-noise scores `0.922`, yet both have `0.000` recall). None of those three feature-space metrics detect memorization: the noisy copier beats even the real training subset on Fréchet (`2.390` vs. `2.694`) and scores `0.905` precision / `0.914` recall, so only `nearest_train_distance` (`0.134` vs. `1.136` for $\beta=1$ VAE) catches copying. Finally, classifier accuracy cannot evaluate an unconditional generator because generated images have no ground-truth labels, and a collapsed model outputting one sharp digit would score near-100% classifier confidence while missing the rest of the distribution.

## Conclusion

* **What the VAE gave that the autoencoder did not:** Although the plain AE reconstructed better (`0.0149`–`0.0153` vs. `0.0259`–`0.0265` MSE), its unregularized latent space left holes where $\mathcal{N}(0, I)$ draws failed (`Fréchet = 74.49`–`108.89`). Regularizing $q(z\vert{}x)$ toward $\mathcal{N}(0, I)$ enabled clean prior sampling (`Fréchet = 11.62`–`13.42` at $\beta=1$, and `7.98`–`9.80` at $\beta=0.5$), while conditioning (`CVAE`) gave `99.95%`–`100.00%` target-digit control.

* **What each metric does and does not capture:** Fréchet distance, precision, recall, and nearest-train distance each catch a distinct failure mode (distributional shift, off-manifold samples, mode collapse, and copying), and Fréchet distance must be compared at identical sample sizes (`n = 359`) and with the same `FeatureNet`.

* **What these measurements establish:** They verify all five Module 07 claims quantitatively on 8×8 digits, while also showing honestly that on this small MLP architecture the VAE latent interpolation path did not beat pixel cross-fade on `FeatureNet` confidence (`0.9173`–`0.9272` vs. `0.9364`), even though it traversed more digit classes (`61`–`64` vs. `50`).

## Known limitations

* **Dataset scale:** Experiments use 8×8 grayscale `sklearn` digits (`1,438` train / `359` test), which are low-resolution and linearly separable enough that a tiny MLP suffices; findings may shift on higher-resolution natural images.

* **Architecture:** Both AE and VAE use a single 64-unit hidden MLP rather than convolutional layers, limiting spatial inductive bias.

* **Evaluation network:** All feature-space metrics depend on a single self-trained `FeatureNet` (`32`-dim ReLU features), which is much smaller than Inception-v3 and sensitive to sample size $N$.

* **Training budget & seeds:** Fixed 200-epoch Adam budget (`lr=2e-3`, `batch_size=64`) across 3 seeds `(0, 1, 2)` without learning-rate scheduling, KL annealing, or deeper hyperparameter tuning.