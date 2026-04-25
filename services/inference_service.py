# services/inference_service.py
# Tek görsel veya batch için tahmin servisi.
# Web arayüzü bu servisi çağırır.

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model.config import (DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                           CLASS_NAMES, CONFIDENCE_THRESHOLD)
from model.model import load_trained_model


# Inference transform (augmentation yok)
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class InferenceService:
    """
    Yüklenmiş model üzerinden tahmin yapan servis.

    Kullanım:
        service = InferenceService('assets/best_model.pth')
        result  = service.predict('path/to/embryo.jpg')
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Eğitilmiş modelin .pth dosya yolu
        """
        self.model      = load_trained_model(model_path)
        self.class_names = CLASS_NAMES

    def predict(self, image_input) -> dict:
        """
        Tek bir embriyo görseli için tahmin yapar.

        Args:
            image_input: Dosya yolu (str) veya PIL.Image nesnesi

        Returns:
            dict: {
                'prediction':    str   — tahmin edilen sınıf (örn: '4AA')
                'confidence':    float — softmax güven skoru (0-1)
                'probabilities': dict  — tüm sınıfların olasılıkları
                'warning':       bool  — güven eşiğinin altında mı?
                'warning_msg':   str   — kullanıcıya gösterilecek uyarı
            }
        """
        # Görsel yükle
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise TypeError('image_input str veya PIL.Image olmalıdır.')

        # Transform ve tahmin
        tensor = _transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(tensor)
            probs  = torch.softmax(output, dim=1)[0]

        pred_idx   = probs.argmax().item()
        confidence = probs[pred_idx].item()
        warning    = confidence < CONFIDENCE_THRESHOLD

        return {
            'prediction'   : self.class_names[pred_idx],
            'confidence'   : round(confidence, 4),
            'probabilities': {
                cls: round(probs[i].item(), 4)
                for i, cls in enumerate(self.class_names)
            },
            'warning'     : warning,
            'warning_msg' : (
                'Bu tahmin düşük güvenilirliktedir, lütfen manuel kontrol yapınız.'
                if warning else ''
            )
        }

    def predict_batch(self, image_paths: list) -> list[dict]:
        """
        Birden fazla görsel için toplu tahmin yapar.

        Args:
            image_paths: Dosya yolları listesi

        Returns:
            list: Her görsel için predict() sonucu
        """
        return [
            {'path': path, **self.predict(path)}
            for path in image_paths
        ]
