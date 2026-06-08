from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _epoch_train_metrics(loss_sum, n_batches, ys, preds):
    train_loss = loss_sum / max(n_batches, 1)
    n = len(ys)
    if n == 0:
        return {
            "train_loss": train_loss,
            "train_accuracy": 0.0,
            "train_macro_f1": 0.0,
            "train_weighted_f1": 0.0,
            "train_micro_f1": 0.0,
            "train_balanced_accuracy": 0.0,
            "train_cohen_kappa": 0.0,
            "train_matthews_corrcoef": 0.0,
            "train_samples": 0,
        }
    acc = 100.0 * sum(a == b for a, b in zip(preds, ys)) / n
    macro_f1 = 100.0 * f1_score(ys, preds, average="macro", zero_division=0)
    weighted_f1 = 100.0 * f1_score(ys, preds, average="weighted", zero_division=0)
    micro_f1 = 100.0 * f1_score(ys, preds, average="micro", zero_division=0)
    bal_acc = 100.0 * balanced_accuracy_score(ys, preds)
    kappa = 100.0 * cohen_kappa_score(ys, preds)
    mcc = float(matthews_corrcoef(ys, preds))
    return {
        "train_loss": train_loss,
        "train_accuracy": acc,
        "train_macro_f1": macro_f1,
        "train_weighted_f1": weighted_f1,
        "train_micro_f1": micro_f1,
        "train_balanced_accuracy": bal_acc,
        "train_cohen_kappa": kappa,
        "train_matthews_corrcoef": mcc,
        "train_samples": n,
    }


def train_model(model, train_loader, epochs, lr, wd, ckpt_path=None):
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if ckpt_path is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    history = []

    for ep in range(epochs):
        model.train()
        loss_sum, ok, n = 0.0, 0, 0
        ys, preds = [], []
        for x, y in tqdm(train_loader, desc=f"{ep+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            pred = out.argmax(1)
            n += y.size(0)
            ok += (pred == y).sum().item()
            ys.extend(y.detach().cpu().tolist())
            preds.extend(pred.detach().cpu().tolist())

        row = {"epoch": ep + 1, **_epoch_train_metrics(loss_sum, len(train_loader), ys, preds)}
        history.append(row)
        print(
            f"ep {ep + 1}: train loss {row['train_loss']:.4f} "
            f"acc {row['train_accuracy']:.1f}% "
            f"macro_f1 {row['train_macro_f1']:.1f}% "
            f"w_f1 {row['train_weighted_f1']:.1f}% "
            f"bal_acc {row['train_balanced_accuracy']:.1f}% "
            f"kappa {row['train_cohen_kappa']:.1f}% "
            f"mcc {row['train_matthews_corrcoef']:.3f}"
        )

    if ckpt_path is not None:
        torch.save(model.state_dict(), ckpt_path)
    return model, history
