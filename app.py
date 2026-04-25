# app.py
# DeepEmbryo Flask Web Uygulaması — Ana giriş noktası
# Servisler: InferenceService, GradCAMService, ReportService (services/ klasöründen)
# Veritabanı: SQLite (database.py)

import io
import csv
import json
import uuid
import os
from flask import (Flask, render_template, request, redirect,
                   url_for, jsonify, Response, flash, abort)
from PIL import Image
from werkzeug.utils import secure_filename

from services.inference_service import InferenceService
from services.gradcam_service   import GradCAMService
from services.report_service    import ReportService
import database as db

# ─────────────────────────────────────────────
# Uygulama Başlatma
# ─────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = 'deepembryo-secret-2024'

# İzin verilen dosya uzantıları
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

# Model servislerini bir kez yükle (uygulama başlangıcında)
inference = InferenceService('assets/best_model.pth')
gradcam   = GradCAMService('assets/best_model.pth')


def allowed_file(filename: str) -> bool:
    """Dosya uzantısının izin verilenler listesinde olup olmadığını kontrol eder."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
# Veritabanı Başlat
# ─────────────────────────────────────────────

with app.app_context():
    db.init_db()


# ─────────────────────────────────────────────
# ANA SAYFA — Tekli Görsel Yükleme
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Ana sayfa: tekli embriyo görseli yükleme formu."""
    return render_template('index.html')


# ─────────────────────────────────────────────
# TEKLİ TAHMİN
# ─────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    """
    Kullanıcının yüklediği bir veya daha fazla görseli tahmin eder.
    InferenceService + GradCAMService kullanır, sonucu DB'ye kaydeder.
    """
    files = request.files.getlist('files')

    if not files or all(f.filename == '' for f in files):
        flash('Lütfen en az bir görsel seçin.', 'error')
        return redirect(url_for('index'))

    batch_id = str(uuid.uuid4())
    report   = ReportService()
    results  = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        filename  = secure_filename(file.filename)

        pred_result = inference.predict(img)
        cam_result  = gradcam.generate(img)

        mode = 'single' if len(files) == 1 else 'batch'

        record = {
            'filename'      : filename,
            'prediction'    : pred_result['prediction'],
            'confidence'    : pred_result['confidence'],
            'warning'       : pred_result['warning'],
            'warning_msg'   : pred_result['warning_msg'],
            'probabilities' : pred_result['probabilities'],
            'region_scores' : cam_result['region_scores'],
            'cam_image_b64' : cam_result['cam_image_b64'],
            'original_b64'  : cam_result['original_b64'],
            'mode'          : mode,
            'batch_id'      : batch_id,
        }

        pred_id = db.insert_prediction(record)
        report.add_result(pred_result)

        results.append({
            'id'           : pred_id,
            'filename'     : filename,
            'prediction'   : pred_result['prediction'],
            'confidence'   : pred_result['confidence'],
            'warning'      : pred_result['warning'],
            'warning_msg'  : pred_result['warning_msg'],
            'original_b64' : cam_result['original_b64'],
            'cam_image_b64': cam_result['cam_image_b64'],
        })

    summary = report.get_summary()

    return render_template(
        'batch_result.html',
        results=results,
        summary=summary,
        batch_id=batch_id
    )


# ─────────────────────────────────────────────
# TAHMİN SONUÇ SAYFASI
# ─────────────────────────────────────────────

@app.route('/result/<int:pred_id>')
def result(pred_id: int):
    """
    Tekli tahmin sonucunu gösterir:
    - Orijinal görsel & Grad-CAM ısı haritası (yan yana)
    - Sınıf olasılık çubukları
    - ICM / TE / Arka Plan bölge skorları
    - Güven uyarısı (confidence < 0.70 ise)
    """
    record = db.get_prediction_by_id(pred_id)
    if record is None:
        abort(404)
    return render_template('result.html', record=record)




# ─────────────────────────────────────────────
# GEÇMİŞ TAHMİNLER
# ─────────────────────────────────────────────

@app.route('/history')
def history():
    """
    Veritabanındaki tüm geçmiş tahminleri listeler.
    Sayfalama: en fazla 200 kayıt gösterilir.
    """
    records = db.get_all_predictions(limit=200)
    return render_template('history.html', records=records)


@app.route('/history/delete/<int:pred_id>', methods=['POST'])
def delete_record(pred_id: int):
    """Geçmiş kaydı siler ve history sayfasına yönlendirir."""
    db.delete_prediction(pred_id)
    flash('Kayıt silindi.', 'success')
    return redirect(url_for('history'))


# ─────────────────────────────────────────────
# EXPORT — CSV / JSON
# ─────────────────────────────────────────────

@app.route('/export/csv')
def export_csv():
    """
    Tüm geçmiş tahminleri CSV formatında indirir.
    Grad-CAM base64 verisi dahil edilmez (boyut yönetimi).
    """
    records = db.get_all_for_export()
    if not records:
        flash('Dışa aktarılacak veri yok.', 'error')
        return redirect(url_for('history'))

    output = io.StringIO()
    # Flat dict oluştur (iç içe dict'leri düzleştir)
    flat_records = []
    for r in records:
        flat = {
            'id'         : r['id'],
            'timestamp'  : r['timestamp'],
            'filename'   : r['filename'],
            'prediction' : r['prediction'],
            'confidence' : r['confidence'],
            'warning'    : r['warning'],
            'mode'       : r['mode'],
        }
        # Olasılık sütunları
        for cls, prob in r.get('probabilities', {}).items():
            flat[f'prob_{cls}'] = prob
        # Bölge skoru sütunları
        for region, score in r.get('region_scores', {}).items():
            flat[f'region_{region}'] = score
        flat_records.append(flat)

    fieldnames = list(flat_records[0].keys()) if flat_records else []
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flat_records)
    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=deepembryo_export.csv'}
    )


@app.route('/export/json')
def export_json():
    """Tüm geçmiş tahminleri JSON formatında indirir."""
    records = db.get_all_for_export()
    payload = {
        'total'  : len(records),
        'results': records
    }
    output = json.dumps(payload, ensure_ascii=False, indent=2)

    return Response(
        output,
        mimetype='application/json',
        headers={'Content-Disposition': 'attachment; filename=deepembryo_export.json'}
    )


# ─────────────────────────────────────────────
# API — Tek sonuç JSON (isteğe bağlı)
# ─────────────────────────────────────────────

@app.route('/api/result/<int:pred_id>')
def api_result(pred_id: int):
    """Belirli bir tahmini JSON olarak döner (API erişimi için)."""
    record = db.get_prediction_by_id(pred_id)
    if record is None:
        return jsonify({'error': 'Kayıt bulunamadı.'}), 404
    return jsonify(record)


# ─────────────────────────────────────────────
# Hata Sayfaları
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# ─────────────────────────────────────────────
# Çalıştır
# ─────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
