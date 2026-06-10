# Fine-Grained Bird Classification

A project for fine-grained bird species classification on the **CUB-200-2011** dataset. The goal is to compare two neural network architectures and two training strategies in terms of species classification accuracy.

## Architectures

| Code | Model | Notes |
|------|-------|-------|
| `cnn` | ResNet50 (`torchvision`) | Pre-trained on ImageNet; head: Dropout + Linear |
| `vit` | ViT-Base/16 (`timm`) | `vit_base_patch16_224`, ImageNet pre-trained weights |

Both models are trained with transfer learning (`PRETRAINED = True` in `config.py`).

## Training strategies

### Flat

Direct classification of all 200 species with a single model. One training phase.

### Hierarchical

Two-stage training that leverages the dataset's taxonomic structure:

1. **Stage 1** — bird family classification.
2. **Stage 2** — replace the classifier head with 200 species and fine-tune with a reduced learning rate (`stage2_lr_mult × learning_rate`).

## Preprocessing and training

- Images scaled to **224×224** px.
- Augmentations (training set): `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`; ImageNet normalization.
- Optimizer: **AdamW** with `CrossEntropyLoss`.

Configuration details — in [`config.py`](config.py).

## Experiments

The main script [`experiments.py`](experiments.py) runs the hyperparameter grid defined in `PARAM_COMBINATIONS`. Each combination consists of **4 training runs**:

```
cnn  × flat
cnn  × hierarchical
vit  × flat
vit  × hierarchical
```

On startup, the script automatically generates label files (`build_labels.py`) from CUB metadata. Results are saved as JSON in `results/`, model weights in `checkpoints/`.

```bash
python experiments.py
```

Individual runs (without the grid) can be launched via the `run_flat()` / `run_hier()` functions in `experiments.py`.

## Metrics

On the test set (`eval.py`), the following are reported, among others:

- accuracy, balanced accuracy
- macro / micro / weighted F1
- Cohen's kappa, Matthews correlation coefficient (MCC)
- top-1, top-3, top-5 accuracy

Training metrics (per epoch) are stored in the `training_per_epoch` field in the result files.

## Data

The project uses the **[CUB-200-2011 dataset on Kaggle](https://www.kaggle.com/datasets/wenewone/cub2002011/data)**. Images are located in `data/images/` (200 subdirectories, one per species). Metadata from the original CUB dataset (labels, train/test split, families) is in the `data/` directory:

| File | Description |
|------|-------------|
| `images.txt` | ID → file path mapping |
| `classes.txt` | 200 species |
| `families.txt` | 38 families |
| `image_class_labels.txt` | species label per image |
| `image_family_labels.txt` | family label per image |
| `train_test_split.txt` | train (5994) / test (5794) split |
| `labels_species.csv` | generated CSV with species labels |
| `labels_family.csv` | generated CSV with family labels |

The `build_labels.py` script builds the CSV files from the metadata above (labels indexed from 0, columns: `image`, `label`, `split`). It is run automatically by `experiments.py`, or can be invoked separately:

```bash
python build_labels.py
```

A detailed description of the CUB file formats — in [`data/README.md`](data/README.md).

## Requirements

- Python 3.10+
- CUDA (optional)

```bash
pip install -r requirements.txt
```

## Project structure

```
├── config.py          # hyperparameters, paths, experiment grid
├── build_labels.py    # CUB label CSV generation
├── datasets.py        # Dataset and DataLoader
├── models.py          # ResNet50 and ViT-Base
├── train.py           # training loop (AdamW)
├── eval.py            # test set evaluation
├── experiments.py     # experiment orchestration
├── data/              # CUB metadata + images (Kaggle)
├── checkpoints/       # model weights (.pth)
└── results/           # experiment results (.json)
```
