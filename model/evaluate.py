# model/evaluate.py
# Test seti metrikleri, confusion matrix ve grafik üreten fonksiyonlar.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from model.config import DEVICE, IMAGENET_MEAN, IMAGENET_STD


def get_predictions(model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader) -> tuple:
    """
    Model tahminlerini ve gerçek etiketleri döner.

    Returns:
        (all_preds, all_labels, all_probs)
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs    = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


def plot_accuracy_loss(history: dict, save_path: str = None):
    """
    Eğitim/validasyon accuracy ve loss grafiklerini çizer.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Eğitim Kaybı')
    axes[0].plot(epochs, history['val_loss'],   'r-o', markersize=4, label='Validasyon Kaybı')
    min_val_idx = history['val_loss'].index(min(history['val_loss']))
    axes[0].axvline(x=min_val_idx + 1, color='green', linestyle='--',
                    alpha=0.7, label=f'En iyi val (epoch {min_val_idx+1})')
    axes[0].set_title('Kayıp (Loss) Eğrisi', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Kayıp')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', markersize=4, label='Eğitim Doğruluğu')
    axes[1].plot(epochs, history['val_acc'],   'r-o', markersize=4, label='Validasyon Doğruluğu')
    axes[1].set_title('Doğruluk (Accuracy) Eğrisi', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Doğruluk')
    axes[1].set_ylim([0, 1.05])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('EfficientNetB3 — Eğitim Performans Grafikleri',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(all_labels: np.ndarray,
                           all_preds: np.ndarray,
                           class_names: list,
                           save_dir: str = None):
    """
    Normal ve normalize edilmiş confusion matrix çizer.
    """
    cm = confusion_matrix(all_labels, all_preds)

    # Normal
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Karışıklık Matrisi)', fontsize=13, fontweight='bold')
    plt.ylabel('Gerçek Sınıf', fontsize=11)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=11)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalize Edilmiş Confusion Matrix', fontsize=13, fontweight='bold')
    plt.ylabel('Gerçek Sınıf', fontsize=11)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=11)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}/confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_f1_per_class(all_labels: np.ndarray,
                      all_preds: np.ndarray,
                      class_names: list,
                      save_dir: str = None) -> dict:
    """
    Sınıf bazında F1-Score bar grafiği çizer ve rapor dict döner.
    """
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )

    print(classification_report(all_labels, all_preds,
                                  target_names=class_names, digits=4))

    per_class_f1 = [report_dict[cls]['f1-score'] for cls in class_names]
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0'][:len(class_names)]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(class_names, per_class_f1, color=colors)
    plt.title('Sınıf Bazında F1-Score', fontsize=13, fontweight='bold')
    plt.ylabel('F1-Score')
    plt.ylim([0, 1.1])
    for bar, val in zip(bars, per_class_f1):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}/f1_per_class.png', dpi=150, bbox_inches='tight')
    plt.show()

    return report_dict
