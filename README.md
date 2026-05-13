# Project 5 — Generative AI Applications (cd001-p5)

Final-capstone deliverable for Udacity AI Mastery Capstone Project 5.
Train and ablate two β-VAE variants on **Fashion-MNIST**: an
unconditional VAE and a class-conditional VAE (CVAE). Both share an
encoder / decoder architecture and a 32-dim latent Gaussian.

## What's in here

```
generative_ai.ipynb     Executable notebook (loads trained checkpoints; runs in seconds)
build_notebook.py       Source of generative_ai.ipynb
models.py               VAE, CVAE, β-VAE loss (importable; re-used by Project 7)
train.py                CLI training driver (15 epochs each on MPS, ~15 min total)
history_*.json          Per-epoch training/test ELBO + recon + KL
best_*.pt               Trained model checkpoints (~2 MB each)
Report.md               Written analysis with APA citations
requirements.txt
README.md / LICENSE / .gitignore
```

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist):
60K 28×28 grayscale clothing images across 10 balanced classes.
Downloaded automatically by torchvision on first run.

## Models

* **VAE** — convolutional encoder
  (Conv 1→32 → Conv 32→64 → flatten → linear) projecting to two
  32-dim heads (mu, logvar); decoder is the mirror image
  (linear → reshape → ConvTranspose → ConvTranspose to logits).
* **CVAE** — same encoder; decoder receives `[z, one_hot(y)]`
  before the first linear layer. Class one-hot is the only
  difference between the two models.

Loss: `β-VAE` ELBO with β = 1.0 (i.e. plain VAE objective).
Optimiser: Adam, lr = 1e-3, batch size 128, 15 epochs.

## Running

```bash
# train both models (~15 min on MPS / Apple Silicon)
python train.py --epochs 15

# run the analysis notebook (loads checkpoints, runs in seconds)
jupyter nbconvert --to notebook --execute --inplace generative_ai.ipynb
```

## License

MIT.
