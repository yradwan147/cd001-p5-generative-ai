"""Train VAE and CVAE on Fashion-MNIST. Writes history + checkpoints to ./.

Usage:
    python train.py --epochs 15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import VAE, CVAE, elbo_loss

SEED = 42
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def pick_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=512, shuffle=False, num_workers=2))


def run(name: str, conditional: bool, epochs: int, lr: float,
        batch_size: int, beta: float) -> dict:
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = pick_device()
    print(f"[{name}] device = {device}")
    model = (CVAE() if conditional else VAE()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_dl, test_dl = loaders(batch_size)
    hist = {"elbo": [], "recon": [], "kl": [], "test_elbo": [], "wall_sec": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        ep_elbo = ep_recon = ep_kl = 0.0
        n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if conditional:
                logits, mu, logvar = model(xb, yb)
            else:
                logits, mu, logvar = model(xb)
            loss, recon, kl = elbo_loss(logits, xb, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            ep_elbo += loss.item() * xb.size(0)
            ep_recon += recon.item() * xb.size(0)
            ep_kl += kl.item() * xb.size(0)
            n += xb.size(0)
        ep_elbo /= n; ep_recon /= n; ep_kl /= n
        # test ELBO
        model.eval()
        test_loss = 0.0; ntest = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                if conditional:
                    logits, mu, logvar = model(xb, yb)
                else:
                    logits, mu, logvar = model(xb)
                loss, _, _ = elbo_loss(logits, xb, mu, logvar, beta=beta)
                test_loss += loss.item() * xb.size(0)
                ntest += xb.size(0)
        test_loss /= ntest
        wall = time.time() - t0
        hist["elbo"].append(ep_elbo); hist["recon"].append(ep_recon)
        hist["kl"].append(ep_kl); hist["test_elbo"].append(test_loss)
        hist["wall_sec"].append(wall)
        print(f"  ep {epoch:>2d}/{epochs}  train_elbo={ep_elbo:.2f} "
              f"(recon={ep_recon:.2f} kl={ep_kl:.2f})  test_elbo={test_loss:.2f}  "
              f"({wall:.0f}s)")
    torch.save(model.state_dict(), Path(__file__).parent / f"best_{name}.pt")
    Path(f"history_{name}.json").write_text(json.dumps(hist, indent=2))
    print(f"[{name}] done, ckpt saved.")
    return hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()
    run("vae", conditional=False, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, beta=args.beta)
    run("cvae", conditional=True, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, beta=args.beta)


if __name__ == "__main__":
    main()
