# services/gradcam_service.py
# Grad-CAM ısı haritası üretme servisi.
# Web arayüzü bu servisi tahmin sonrası çağırır.

import io
import base64
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model.config import (DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                           CLASS_NAMES)
from model.model import load_trained_model


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class GradCAMService:
    """
    Grad-CAM ısı haritası üreten servis.

    Kullanım:
        service = GradCAMService('assets/best_model.pth')
        result  = service.generate('path/to/embryo.jpg', target_class='4AA')
    """

    def __init__(self, model_path: str):
        self.model       = load_trained_model(model_path)
        self.class_names = CLASS_NAMES
        # EfficientNetB3'ün son conv bloğu hedef katman
        self.target_layer = [self.model.features[-1]]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layer)

    def generate(self, image_input, target_class: str = None) -> dict:
        """
        Görsel için Grad-CAM ısı haritası üretir.

        Args:
            image_input:  Dosya yolu (str) veya PIL.Image
            target_class: Hedef sınıf adı (None ise model tahmini kullanılır)

        Returns:
            dict: {
                'cam_image_b64':   str  — base64 PNG (web için)
                'cam_image_pil':   PIL.Image — doğrudan kullanım için
                'original_b64':    str  — orijinal görsel base64
                'predicted_class': str
                'confidence':      float
                'region_scores':   dict — ICM/TE/Arka Plan aktivasyon skorları
            }
        """
        # Görsel yükle
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        else:
            pil_img = image_input.convert('RGB')

        pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
        tensor  = _transform(pil_img).unsqueeze(0).to(DEVICE)

        # Model tahmini
        with torch.no_grad():
            output = self.model(tensor)
            probs  = torch.softmax(output, dim=1)[0]

        pred_idx   = probs.argmax().item()
        confidence = probs[pred_idx].item()

        # Hedef sınıf
        if target_class is not None and target_class in self.class_names:
            target_idx = self.class_names.index(target_class)
        else:
            target_idx = pred_idx

        # Grad-CAM hesapla
        targets      = [ClassifierOutputTarget(target_idx)]
        grayscale_cam = self.cam(input_tensor=tensor, targets=targets)[0]

        # Görseli denormalize et
        img_np = tensor[0].cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1).astype(np.float32)

        # Isı haritası overlay
        cam_array = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        cam_pil   = Image.fromarray(cam_array)

        # Bölgesel aktivasyon analizi (ICM / TE / Arka Plan)
        region_scores = self._compute_region_scores(grayscale_cam)

        return {
            'cam_image_b64'  : self._pil_to_b64(cam_pil),
            'cam_image_pil'  : cam_pil,
            'original_b64'   : self._pil_to_b64(pil_img),
            'predicted_class': self.class_names[pred_idx],
            'confidence'     : round(confidence, 4),
            'region_scores'  : region_scores
        }

    def _compute_region_scores(self, grayscale_cam: np.ndarray) -> dict:
        """
        Embriyoyu 3 bölgeye bölerek aktivasyon skorlarını hesaplar:
          - ICM (Merkez): İç hücre kütlesi
          - TE (Kenar Halkası): Trofektoderm
          - Arka Plan: Blastosist dışı
        """
        h, w  = grayscale_cam.shape
        cy, cx = h // 2, w // 2
        r_icm = min(h, w) // 5
        r_te  = min(h, w) // 2

        Y, X = np.ogrid[:h, :w]
        dist  = np.sqrt((X - cx)**2 + (Y - cy)**2)

        mask_icm = dist < r_icm
        mask_te  = (dist >= r_icm) & (dist < r_te)
        mask_bg  = dist >= r_te

        scores = {
            'ICM (Merkez)'     : round(float(grayscale_cam[mask_icm].mean()), 4),
            'TE (Kenar Halkası)': round(float(grayscale_cam[mask_te].mean()),  4),
            'Arka Plan'        : round(float(grayscale_cam[mask_bg].mean()),   4)
        }
        scores['dominant'] = max(
            {k: v for k, v in scores.items() if k != 'dominant'},
            key=lambda k: scores[k]
        )
        return scores

    @staticmethod
    def _pil_to_b64(pil_img: Image.Image) -> str:
        """PIL görselini base64 string'e çevirir (web için)."""
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
