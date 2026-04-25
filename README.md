# DeepEmbryo
**Yapay Zeka Destekli IVF Embriyo Kalite Analiz Sistemi**

EfficientNetB3 tabanlı derin öğrenme modeli ile IVF tedavisindeki blastosist embriyolarının Gardner skalasına göre otomatik sınıflandırılması.

---

## Model Performansı

| Sınıf | F1-Score | Açıklama |
|-------|----------|----------|
| 3AA | 0.769 | İyi kalite, 3. gün genişleme |
| 3CC | 0.667 | Düşük kalite — morfolojik benzerlik nedeniyle en zor sınıf |
| 4AA | 0.800 | İyi kalite, 4. gün genişleme |
| Cleavage | 1.000 | Erken evre embriyo |
| **Weighted Avg** | **0.801** | |
| **Test Accuracy** | **%80.77** | |

---

##Temel Özellikler

- **Tekli ve Çoklu (Batch) Analiz**
- **Gardner Skalası Sınıflandırması:** `3AA`, `3CC`, `4AA`, `Cleavage`
- **Açıklanabilir Yapay Zeka (Grad-CAM):** Isı haritası görselleştirme
- **Morfolojik Bölge Skorlaması:** ICM ve TE bölge analizi
- **Düşük Güven Uyarısı:** %70 altında manuel doğrulama uyarısı
- **Raporlama:** CSV ve JSON formatında dışa aktarım

---

## Kullanılan Teknolojiler

- **Backend:** Python, Flask, SQLite
- **Yapay Zeka:** PyTorch, torchvision, pytorch-grad-cam
- **Model:** EfficientNetB3 (Transfer Learning)
- **Frontend:** HTML5, CSS3, JavaScript ES6+

---

## Proje Yapısı

```text
├── app.py                   # Flask ana uygulaması
├── database.py              # SQLite veritabanı işlemleri
├── main.py                  # FastAPI (API endpoint'leri)
├── requirements.txt
├── assets/
│   ├── best_model.pth       # Eğitilmiş model (git'e dahil değil)
│   └── model_info.json
├── model/
│   ├── config.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
├── services/
│   ├── inference_service.py
│   ├── gradcam_service.py
│   └── report_service.py
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    ├── base.html
    ├── index.html
    ├── batch_result.html
    ├── result.html
    ├── history.html
    ├── 404.html
    └── 500.html
```

---

## Kurulum

```bash
pip install -r requirements.txt
```

`best_model.pth` dosyasını `assets/` klasörüne ekle.

> **Model dosyasını indir:** [best_model.pth (41MB)](https://drive.google.com/file/d/15iAFTBtFB0FYFVkSjZleNtESMTi4kQtM/view?usp=sharing)

---

## Çalıştırma

```bash
# Web arayüzü
python app.py
# → http://localhost:5000

# API
uvicorn main:app --reload
# → http://127.0.0.1:8000
```

---

## Referanslar

- Gardner & Schoolcraft (1999). *Culture and transfer of human blastocysts.*
- Tan et al. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.*