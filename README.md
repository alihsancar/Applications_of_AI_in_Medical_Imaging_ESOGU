# DeepEmbryo 
**Yapay Zeka Destekli IVF Embriyo Kalite Analiz Sistemi**

Bu proje, Tüp Bebek (IVF) tedavisi gören hastalar için kritik öneme sahip olan 5. gün (blastosist) embriyolarının kalitesini, **PyTorch** tabanlı bir Derin Öğrenme modeli (EfficientNetB3) ile objektif ve otomatik olarak değerlendiren bir **Flask** web uygulamasıdır.

Embriyologlar tarafından mikroskop görüntüleri üzerinden yapılan subjektif değerlendirmelerin doğruluğunu artırmak ve Açıklanabilir Yapay Zeka (XAI) yöntemleriyle (Grad-CAM) model kararlarını şeffaflaştırmak hedeflenmiştir.

---

##  Temel Özellikler

- **Tekli ve Çoklu (Batch) Analiz:** İster bir, ister birden fazla embriyo görselini aynı ekrandan yükleyerek saniyeler içinde analiz edebilirsiniz.
- **Gardner Skalası Sınıflandırması:** Embriyolar `3AA`, `3CC`, `4AA` ve `Cleavage` (bölünme evresi) sınıflarından birine yüksek doğrulukla yerleştirilir.
- **Açıklanabilir Yapay Zeka (Grad-CAM):** Tahminlerin hangi morfolojik bölgelere dayandığını gösteren renkli ısı haritaları üretilir.
- **Morfolojik Bölge Skorlaması:** İç Hücre Kütlesi (ICM) ve Trofektoderm (TE) bölgelerinin modele olan etkisi bölgesel olarak hesaplanıp sunulur.
- **Düşük Güven Uyarısı:** Modelin tahmindeki güvenilirlik oranı `%70`'in altındaysa sistem embriyoloğa manuel doğrulama uyarısı verir.
- **Veritabanı ve Raporlama:** Yapılan tüm analizler (görsellerle birlikte) SQLite veritabanına kaydedilir. İstenildiği zaman `CSV` veya `JSON` formatında dışa aktarılabilir.

---

## Kullanılan Teknolojiler

- **Backend:** Python, Flask, SQLite
- **Yapay Zeka & Görüntü İşleme:** PyTorch, torchvision, OpenCV, PIL, pytorch-grad-cam
- **Model Mimarisi:** EfficientNetB3
- **Frontend:** HTML5, CSS3 (Vanilla - Dark Glassmorphism Tema), JavaScript (ES6+)

---

## Proje Yapısı

```text
├── app.py                   # Flask ana uygulaması ve rotalar (predict, history, export vs.)
├── database.py              # SQLite veritabanı işlemleri (CRUD)
├── requirements.txt         # Proje bağımlılıkları listesi
├── assets/
│   ├── best_model.pth       # Eğitilmiş PyTorch model ağırlıkları (Kullanıcı ekler)
│   └── model_info.json      # Model ve sınıflara ait konfigürasyon detayları
├── model/                   # PyTorch model tanımları ve eğitim kodları
│   ├── config.py
│   ├── model.py
│   └── train.py
├── services/                # İş mantığı servisleri
│   ├── gradcam_service.py   # Grad-CAM ısı haritası ve bölge aktivasyonu üreten servis
│   ├── inference_service.py # Görüntü ön işleme ve model tahmin servisi
│   └── report_service.py    # Çoklu sonuçların özet istatistiklerini hesaplayan servis
├── static/
│   ├── css/style.css        # Uygulamanın modern karanlık teması
│   └── js/main.js           # Drag & Drop, yükleme ekranı ve dinamik UI etkileşimleri
└── templates/               # Flask HTML şablonları
    ├── base.html            # Ana iskelet ve navigasyon
    ├── index.html           # Ana sayfa, birleşik çoklu/tekli yükleme alanı
    ├── batch_result.html    # Çoklu yükleme sonucunda oluşan grid tasarımlı sonuç listesi
    ├── result.html          # Geçmişten erişilen tekli detay sayfası
    ├── history.html         # Veritabanı geçmiş listesi ve raporlama işlemleri
    ├── 404.html             # Sayfa bulunamadı hatası
    └── 500.html             # Sunucu hatası
```

---

## Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8 veya üzeri
- Eğitilmiş model dosyası (`assets/best_model.pth`) projede bulunmalıdır.

### Adımlar

1. **Bağımlılıkları Yükleyin:**
   Terminal üzerinden proje klasörüne gidin ve gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. **Uygulamayı Başlatın:**
   Flask uygulamasını çalıştırın:
   ```bash
   python app.py
   ```

3. **Arayüze Erişin:**
   Tarayıcınızı açın ve aşağıdaki adrese gidin:
   ```text
   http://localhost:5000
   ```

---

## Kullanım Senaryoları

1. **Yeni Analiz:** `http://localhost:5000/` adresinde açılan ekrana embriyo görselini (veya birden fazla görseli) sürükleyip bırakın ve "Analizi Başlat" butonuna tıklayın.
2. **Sonuçları İnceleme:** İşlem bittiğinde orijinal görsel ile yan yana Grad-CAM ısı haritasını göreceksiniz. Güven skoru, sınıf olasılıkları ve bölgesel analizler kart üzerinde yer alır.
3. **Geçmiş Görüntüleme:** Menüden `Geçmiş` sekmesine geçerek daha önceki tahminlerinizi filtreleyebilir, tekil detaylarına bakabilir veya kayıtları silebilirsiniz.
4. **Dışa Aktarma:** Geçmiş sayfasından "CSV İndir" veya "JSON İndir" butonları ile tüm tahmin verilerini raporlayabilirsiniz.