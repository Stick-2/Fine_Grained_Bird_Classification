import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)

from train import device


def _topk_accuracy(logits, target, topk=(1, 3, 5)):
    """Returns dict top{k}_accuracy in percent for each k."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    out = {}
    for k in topk:
        out[k] = correct[:k].reshape(-1).float().sum().item()
    return out, batch_size


def evaluate(model, test_loader, topk=(1, 3, 5)):
    model = model.to(device).eval()
    preds, ys = [], []
    topk_totals = {k: 0.0 for k in topk}
    n_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_dev = y.to(device)
            logits = model(x)
            preds += logits.argmax(1).cpu().tolist()
            ys += y.tolist()
            inc, bs = _topk_accuracy(logits, y_dev, topk=topk)
            n_total += bs
            for k in topk:
                topk_totals[k] += inc[k]

    n = len(ys)
    acc = 100.0 * sum(a == b for a, b in zip(preds, ys)) / n
    macro_f1 = 100.0 * f1_score(ys, preds, average="macro", zero_division=0)
    micro_f1 = 100.0 * f1_score(ys, preds, average="micro", zero_division=0)
    weighted_f1 = 100.0 * f1_score(ys, preds, average="weighted", zero_division=0)
    bal_acc = 100.0 * balanced_accuracy_score(ys, preds)
    kappa = 100.0 * cohen_kappa_score(ys, preds)
    mcc = float(matthews_corrcoef(ys, preds))

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "cohen_kappa": kappa,
        "matthews_corrcoef": mcc,
        "test_samples": n,
    }
    for k in topk:
        metrics[f"top{k}_accuracy"] = 100.0 * topk_totals[k] / max(n_total, 1)
    return metrics
