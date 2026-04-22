# model/train.py
# Model eğitimi ve early stopping döngüsü.

import torch
import torch.nn as nn
import torch.optim as optim
from model.config import (DEVICE, NUM_EPOCHS, EARLY_STOP_PATIENCE,
                           UNFREEZE_EPOCH, LEARNING_RATE, WEIGHT_DECAY,
                           LABEL_SMOOTHING, CLASS_WEIGHTS)
from model.model import unfreeze_all


def train_one_epoch(model: nn.Module,
                    loader: torch.utils.data.DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer) -> tuple[float, float]:
    """
    Tek bir epoch için eğitim döngüsü.

    Returns:
        (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model: nn.Module,
             loader: torch.utils.data.DataLoader,
             criterion: nn.Module) -> tuple[float, float]:
    """
    Validasyon / test değerlendirme döngüsü.

    Returns:
        (loss, accuracy)
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs      = model(imgs)
            loss         = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, predicted  = outputs.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          save_path: str) -> dict:
    """
    Tam eğitim döngüsü. Early stopping ve unfreeze stratejisi içerir.

    Args:
        model:        Eğitilecek model
        train_loader: Eğitim DataLoader
        val_loader:   Validasyon DataLoader
        save_path:    En iyi modelin kaydedileceği .pth yolu

    Returns:
        history: {train_loss, val_loss, train_acc, val_acc} listeleri
    """
    # Sınıf ağırlıklı kayıp fonksiyonu
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                     label_smoothing=LABEL_SMOOTHING)

    # Sadece eğitilebilir parametreler için optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {'train_loss': [], 'val_loss': [],
               'train_acc' : [], 'val_acc' : []}

    best_val_loss  = float('inf')
    patience_count = 0

    print(f'{"Epoch":<8} {"Train Loss":<12} {"Train Acc":<12} '
          f'{"Val Loss":<12} {"Val Acc":<12} {"LR"}')
    print('-' * 65)

    for epoch in range(1, NUM_EPOCHS + 1):

        # UNFREEZE: belirli epoch'tan sonra tüm katmanları aç
        if epoch == UNFREEZE_EPOCH:
            model = unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(),
                                     lr=1e-4, weight_decay=WEIGHT_DECAY)
            print(f'\n Epoch {epoch}: Tüm katmanlar açıldı (full fine-tune)\n')

        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                 criterion, optimizer)
        val_loss, val_acc     = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'{epoch:<8} {train_loss:<12.4f} {train_acc:<12.4f} '
              f'{val_loss:<12.4f} {val_acc:<12.4f} {current_lr:.2e}')

        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f'   💾 Yeni en iyi model kaydedildi! (val_loss: {val_loss:.4f})')
        else:
            patience_count += 1

        # Early Stopping
        if patience_count >= EARLY_STOP_PATIENCE:
            print(f'\n⏹️  Early Stopping: {epoch}. epoch\'ta durduruldu.')
            break

    print(f'\n✅ Eğitim tamamlandı! En iyi val_loss: {best_val_loss:.4f}')
    return history
