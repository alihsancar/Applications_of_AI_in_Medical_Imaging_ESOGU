# model/dataset.py
# Veri yükleme, augmentation ve DataLoader hazırlama.

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from model.config import (IMG_SIZE, BATCH_SIZE, SEED,
                           IMAGENET_MEAN, IMAGENET_STD)


# --- Transform Tanımları ---

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Inference için (tek görsel, augmentation yok)
inference_transform = val_test_transform


class TransformSubset(Dataset):
    """
    ImageFolder'dan gelen veriyi belirtilen transform ile saran Dataset.
    Train/Val/Test split'lerinde farklı transform uygulamak için kullanılır.
    """
    def __init__(self, dataset: datasets.ImageFolder,
                 indices: list,
                 transform: transforms.Compose):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        img = self.transform(img)
        return img, label


def load_datasets(data_dir: str) -> dict:
    """
    Verilen klasörden veriyi yükler ve %70/%15/%15 olarak böler.

    Args:
        data_dir: Sınıf klasörlerini içeren ana dizin
                  Örn: /data/  →  /data/3AA/, /data/3CC/, ...

    Returns:
        dict: {
            'train': TransformSubset,
            'val':   TransformSubset,
            'test':  TransformSubset,
            'class_names': list,
            'num_classes': int,
            'full_dataset': ImageFolder,
            'train_idx': list,
            'val_idx': list,
            'test_idx': list
        }
    """
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names  = full_dataset.classes
    num_classes  = len(class_names)

    all_indices = list(range(len(full_dataset)))
    all_labels  = [full_dataset.targets[i] for i in all_indices]

    # Stratified split: %70 train, %15 val, %15 test
    train_idx, temp_idx = train_test_split(
        all_indices, test_size=0.30, stratify=all_labels, random_state=SEED
    )
    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=SEED
    )

    train_dataset = TransformSubset(full_dataset, train_idx, train_transform)
    val_dataset   = TransformSubset(full_dataset, val_idx,   val_test_transform)
    test_dataset  = TransformSubset(full_dataset, test_idx,  val_test_transform)

    return {
        'train'       : train_dataset,
        'val'         : val_dataset,
        'test'        : test_dataset,
        'class_names' : class_names,
        'num_classes' : num_classes,
        'full_dataset': full_dataset,
        'train_idx'   : train_idx,
        'val_idx'     : val_idx,
        'test_idx'    : test_idx
    }


def build_dataloaders(data_dir: str) -> dict:
    """
    Veriyi yükler, WeightedRandomSampler uygular ve DataLoader'ları döner.

    Args:
        data_dir: Veri seti ana dizini

    Returns:
        dict: {
            'train_loader', 'val_loader', 'test_loader',
            'class_names', 'num_classes', 'test_dataset'
        }
    """
    ds = load_datasets(data_dir)

    # Sınıf ağırlıkları (az örnekli sınıfı daha sık göster)
    class_counts   = [len(os.listdir(os.path.join(data_dir, c)))
                      for c in ds['class_names']]
    weights        = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [weights[ds['full_dataset'].targets[i]]
                      for i in ds['train_idx']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(ds['train'], batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ds['val'],   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(ds['test'],  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    return {
        'train_loader': train_loader,
        'val_loader'  : val_loader,
        'test_loader' : test_loader,
        'class_names' : ds['class_names'],
        'num_classes' : ds['num_classes'],
        'test_dataset': ds['test']
    }
