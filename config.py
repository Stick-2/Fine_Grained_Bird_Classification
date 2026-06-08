from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Raw CUB metadata files (images.txt, bounding_boxes.txt, etc.)
CUB_ROOT = DATA_DIR
# Directory with training images (cropped/original depending on your setup).
IMAGES_DIR_NAME = "images"
# Backward-compatible fallback location used by older scripts.
IMAGES_FALLBACK_DIR_NAME = "images_cropped"

DATA_ROOT = str(DATA_DIR)
LABELS_SPECIES = str(DATA_DIR / "labels_species.csv")
LABELS_FAMILY = str(DATA_DIR / "labels_family.csv")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

EPOCHS_FLAT = 80
EPOCHS_FAMILY = 20
EPOCHS_SPECIES_FINETUNE = 60
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

CHECKPOINT_DIR = str(BASE_DIR / "checkpoints")
RESULTS_DIR = str(BASE_DIR / "results")

PRETRAINED = True
DROPOUT = 0.5

TRAIN_CROP_PADDING = 32
COLOR_JITTER = (0.4, 0.4, 0.4, 0.1)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Siatka eksperymentów: każdy wpis nadpisuje tylko podane klucze (reszta jak
# wyżej). Pusta lista = jeden domyślny przebieg bez nadpisań.
#
# Jedna kombinacja = 4 treningi (cnn/vit × flat/hierarchical) — dobór celowy,
# nie pełny iloczyn (byłby setkami uruchomień).
# ---------------------------------------------------------------------------
PARAM_COMBINATIONS = [
    # punkt odniesienia (jak LEARNING_RATE / WEIGHT_DECAY / … z góry pliku)
    #{},
    # częstość uczenia (AdamW + transfer)
    #{"learning_rate": 3e-4, "weight_decay": 1e-4},
    #{"learning_rate": 1e-3, "weight_decay": 1e-4},
    #{"learning_rate": 3e-3, "weight_decay": 1e-4},
    # regularyzacja L2
    {"learning_rate": 3e-4, "weight_decay": 1e-4},
    #{"learning_rate": 1e-3, "weight_decay": 1e-2},
    #{"learning_rate": 3e-4, "weight_decay": 0.0},
    # głowa CNN (ViT i tak przechodzi z tym samym zestawem — porównanie arch)
    #{"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.3},
    #{"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.7},
    # wielkość batcha (szum gradientu vs stabilność)
    #{"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 16},
    #{"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64},
    # hierarchia: LR drugiego etapu względem pierwszego (tylko wpływ na stage2)
    #{"learning_rate": 1e-3, "weight_decay": 1e-4, "stage2_lr_mult": 0.03},
    #{"learning_rate": 1e-3, "weight_decay": 1e-4, "stage2_lr_mult": 0.3},
]
