import json
import re
from pathlib import Path

import build_labels
import config
from datasets import get_dataloaders
from eval import evaluate
from models import build_model, replace_head
from train import train_model


def _hparam_defaults():
    return {
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "dropout": config.DROPOUT,
        "batch_size": config.BATCH_SIZE,
        "epochs_flat": config.EPOCHS_FLAT,
        "epochs_family": config.EPOCHS_FAMILY,
        "epochs_species": config.EPOCHS_SPECIES_FINETUNE,
        "stage2_lr_mult": 0.1,
        "train_crop_padding": config.TRAIN_CROP_PADDING,
        "color_jitter": tuple(config.COLOR_JITTER),
    }


def _dl_kwargs(hp):
    return dict(
        batch_size=hp["batch_size"],
        train_crop_padding=hp["train_crop_padding"],
        color_jitter=hp["color_jitter"],
    )


def run_flat_with_hparams(arch, hp, *, ckpt_path=None):
    hp = {**_hparam_defaults(), **hp}
    tr, te, ncls = get_dataloaders(
        config.DATA_ROOT, config.LABELS_SPECIES, **_dl_kwargs(hp)
    )
    m = build_model(arch, ncls, dropout=hp["dropout"])
    _, hist = train_model(
        m,
        tr,
        hp["epochs_flat"],
        hp["learning_rate"],
        hp["weight_decay"],
        ckpt_path,
    )
    test_metrics = evaluate(m, te)
    return {
        "architecture": arch,
        "strategy": "flat",
        "hyperparameters": hp,
        "training_per_epoch": hist,
        "test": test_metrics,
    }


def run_hier_with_hparams(arch, hp, *, ckpt_stage1_path=None, ckpt_final_path=None):
    hp = {**_hparam_defaults(), **hp}
    lr2 = hp["learning_rate"] * hp["stage2_lr_mult"]
    tr_f, te_f, n_fam = get_dataloaders(
        config.DATA_ROOT, config.LABELS_FAMILY, **_dl_kwargs(hp)
    )
    m = build_model(arch, n_fam, dropout=hp["dropout"])
    _, hist_f = train_model(
        m,
        tr_f,
        hp["epochs_family"],
        hp["learning_rate"],
        hp["weight_decay"],
        ckpt_stage1_path,
    )
    tr_s, te_s, n_sp = get_dataloaders(
        config.DATA_ROOT, config.LABELS_SPECIES, **_dl_kwargs(hp)
    )
    replace_head(m, n_sp, arch, dropout=hp["dropout"])
    _, hist_s = train_model(
        m,
        tr_s,
        hp["epochs_species"],
        lr2,
        hp["weight_decay"],
        ckpt_final_path,
    )
    test_metrics = evaluate(m, te_s)
    return {
        "architecture": arch,
        "strategy": "hierarchical",
        "hyperparameters": hp,
        "training_per_epoch": {
            "stage1_family": hist_f,
            "stage2_species_finetune": hist_s,
        },
        "test": test_metrics,
    }


def _strip_run(out: dict) -> dict:
    return {
        "training_per_epoch": out.get("training_per_epoch"),
        "test": out.get("test"),
    }


def _filename_for_combo(index: int, override: dict) -> str:
    if not override:
        name = "default"
    else:
        parts = []
        for k in sorted(override.keys()):
            v = override[k]
            if isinstance(v, float):
                parts.append(f"{k}-{v:g}".replace(".", "p"))
            else:
                parts.append(f"{k}-{v}")
        raw = "_".join(parts)
        raw = re.sub(r'[^0-9A-Za-z._-]+', "_", raw)[:120]
        name = raw or "combo"
    return f"grid_{index:03d}_{name}.json"


def run_flat(arch):
    print(f"\nflat {arch}")
    hp = _hparam_defaults()
    out = run_flat_with_hparams(
        arch,
        hp,
        ckpt_path=f"{config.CHECKPOINT_DIR}/{arch}_flat.pth",
    )
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/{arch}_flat.json", "w") as f:
        json.dump(out, f, indent=2)
    print(out)
    return out


def run_hier(arch):
    print(f"\nhierarchical {arch}")
    hp = _hparam_defaults()
    out = run_hier_with_hparams(
        arch,
        hp,
        ckpt_stage1_path=f"{config.CHECKPOINT_DIR}/{arch}_hier_stage1.pth",
        ckpt_final_path=f"{config.CHECKPOINT_DIR}/{arch}_hierarchical.pth",
    )
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/{arch}_hierarchical.json", "w") as f:
        json.dump(out, f, indent=2)
    print(out)
    return out


def _run_one_combo(override: dict, index: int, results_dir: Path, ckpt_dir: Path) -> None:
    hp = {**_hparam_defaults(), **override}
    fname = _filename_for_combo(index, override)
    out_path = results_dir / fname

    comparison = {}
    for arch in ("cnn", "vit"):
        print(f"\n--- combo {index} {fname} | {arch} ---", flush=True)
        comparison[arch] = {
            "flat": _strip_run(
                run_flat_with_hparams(
                    arch,
                    hp,
                    ckpt_path=str(ckpt_dir / f"grid_{index:03d}_{arch}_flat.pth"),
                )
            ),
            "hierarchical": _strip_run(
                run_hier_with_hparams(
                    arch,
                    hp,
                    ckpt_stage1_path=str(ckpt_dir / f"grid_{index:03d}_{arch}_hier_stage1.pth"),
                    ckpt_final_path=str(ckpt_dir / f"grid_{index:03d}_{arch}_hierarchical.pth"),
                )
            ),
        }

    payload = {
        "combo_index": index,
        "override": override,
        "hyperparameters": hp,
        "comparison": comparison,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nZapisano {out_path}", flush=True)


if __name__ == "__main__":
    build_labels.main()
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(config.CHECKPOINT_DIR)

    combos = getattr(config, "PARAM_COMBINATIONS", None)
    if combos is None:
        combos = []
    combos = list(combos)
    if not combos:
        combos = [{}]

    for i, override in enumerate(combos):
        _run_one_combo(override, i, results_dir, ckpt_dir)
