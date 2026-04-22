# model/config.py
# Tüm sabitler ve hiperparametreler burada tanımlanır.

import torch

# --- Veri ---
IMG_SIZE    = 300
BATCH_SIZE  = 16
SEED        = 42

# ImageNet normalizasyon değerleri (Transfer Learning için zorunlu)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- Sınıflar ---
CLASS_NAMES = ['3AA', '3CC', '4AA', 'Cleavage']
NUM_CLASSES = len(CLASS_NAMES)

# --- Model ---
DROPOUT_RATE = 0.6

# --- Eğitim ---
NUM_EPOCHS           = 60
EARLY_STOP_PATIENCE  = 10
UNFREEZE_EPOCH       = 25
LEARNING_RATE        = 1e-3
WEIGHT_DECAY         = 1e-4
LABEL_SMOOTHING      = 0.1

# Sınıf ağırlıkları [3AA, 3CC, 4AA, Cleavage]
# 3CC ve 4AA'ya ekstra ağırlık verildi (az veri & zor sınıf)
CLASS_WEIGHTS = [1.0, 2.0, 1.5, 1.0]

# --- XAI ---
CONFIDENCE_THRESHOLD = 0.70   # Altında uyarı göster

# --- Cihaz ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
