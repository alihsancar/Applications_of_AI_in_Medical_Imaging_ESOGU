# model/model.py
# EfficientNetB3 model tanımı ve yükleme fonksiyonları.

import torch
import torch.nn as nn
from torchvision import models
from model.config import DEVICE, DROPOUT_RATE, NUM_CLASSES


def build_efficientnetb3(num_classes: int = NUM_CLASSES,
                          dropout_rate: float = DROPOUT_RATE,
                          pretrained: bool = True) -> nn.Module:
    """
    ImageNet üzerinde eğitilmiş EfficientNetB3 yükler.
    Classifier katmanı verilen sınıf sayısına göre değiştirilir.

    Args:
        num_classes:   Çıkış sınıf sayısı
        dropout_rate:  Classifier dropout oranı
        pretrained:    ImageNet ağırlıklarını kullan

    Returns:
        model: DEVICE'a taşınmış PyTorch modeli
    """
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    # Feature extractor katmanlarını dondur (ilk eğitimde)
    for param in model.features.parameters():
        param.requires_grad = False

    # Son classifier'ı sınıf sayısına göre değiştir
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(in_features, num_classes)
        # Not: Softmax PyTorch'ta CrossEntropyLoss içinde zaten var
    )

    return model.to(DEVICE)


def load_trained_model(model_path: str,
                        num_classes: int = NUM_CLASSES,
                        dropout_rate: float = DROPOUT_RATE) -> nn.Module:
    """
    Kaydedilmiş model ağırlıklarını yükler (inference için).

    Args:
        model_path:    .pth dosyasının yolu
        num_classes:   Sınıf sayısı
        dropout_rate:  Dropout oranı

    Returns:
        model: Eval modunda yüklenmiş model
    """
    model = build_efficientnetb3(num_classes=num_classes,
                                  dropout_rate=dropout_rate,
                                  pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """
    Tüm katmanların gradyanını açar (full fine-tune için).

    Args:
        model: Kısmen dondurulmuş model

    Returns:
        model: Tüm parametreleri eğitilebilir model
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_model_summary(model: nn.Module) -> dict:
    """
    Model parametre özetini döner.

    Returns:
        dict: total, trainable, frozen parametre sayıları
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params'    : total,
        'trainable_params': trainable,
        'frozen_params'   : total - trainable
    }
