# database.py
# SQLite entegrasyonu — tahmin geçmişi yönetimi

import sqlite3
import json
from datetime import datetime

DB_PATH = 'deepembryo.db'


def get_connection():
    """Veritabanı bağlantısı döner."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Sütun ismine göre erişim
    return conn


def init_db():
    """
    Veritabanını ve tabloyu oluşturur.
    Uygulama başlarken çağrılır.
    """
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            filename        TEXT    NOT NULL,
            prediction      TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            warning         INTEGER NOT NULL DEFAULT 0,
            warning_msg     TEXT    DEFAULT '',
            probabilities   TEXT    DEFAULT '{}',
            region_scores   TEXT    DEFAULT '{}',
            cam_image_b64   TEXT    DEFAULT '',
            original_b64    TEXT    DEFAULT '',
            mode            TEXT    DEFAULT 'single',
            batch_id        TEXT    DEFAULT ''
        )
    ''')
    conn.commit()
    conn.close()


def insert_prediction(data: dict) -> int:
    """
    Bir tahmin sonucunu veritabanına ekler.

    Args:
        data: {
            filename, prediction, confidence, warning, warning_msg,
            probabilities, region_scores, cam_image_b64, original_b64,
            mode ('single'|'batch'), batch_id (opsiyonel)
        }
    Returns:
        Eklenen kaydın ID'si
    """
    conn = get_connection()
    cursor = conn.execute('''
        INSERT INTO predictions
            (timestamp, filename, prediction, confidence, warning, warning_msg,
             probabilities, region_scores, cam_image_b64, original_b64, mode, batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data.get('filename', ''),
        data.get('prediction', ''),
        data.get('confidence', 0.0),
        1 if data.get('warning') else 0,
        data.get('warning_msg', ''),
        json.dumps(data.get('probabilities', {}), ensure_ascii=False),
        json.dumps(data.get('region_scores', {}), ensure_ascii=False),
        data.get('cam_image_b64', ''),
        data.get('original_b64', ''),
        data.get('mode', 'single'),
        data.get('batch_id', ''),
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_all_predictions(limit: int = 200) -> list:
    """
    Tüm tahminleri yeniden eskiye sıralar (Grad-CAM base64 hariç — listede gereksiz).

    Args:
        limit: Maksimum kayıt sayısı
    Returns:
        dict listesi
    """
    conn = get_connection()
    rows = conn.execute('''
        SELECT id, timestamp, filename, prediction, confidence,
               warning, warning_msg, mode, batch_id
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction_by_id(pred_id: int) -> dict | None:
    """
    ID'ye göre tek tahmin kaydını döner (Grad-CAM dahil).

    Args:
        pred_id: Tahmin ID'si
    Returns:
        dict veya None
    """
    conn = get_connection()
    row = conn.execute(
        'SELECT * FROM predictions WHERE id = ?', (pred_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    # JSON alanları parse et
    d['probabilities'] = json.loads(d.get('probabilities') or '{}')
    d['region_scores'] = json.loads(d.get('region_scores') or '{}')
    return d


def delete_prediction(pred_id: int) -> bool:
    """
    Belirtilen ID'li kaydı siler.

    Returns:
        True — silindi, False — bulunamadı
    """
    conn = get_connection()
    cursor = conn.execute('DELETE FROM predictions WHERE id = ?', (pred_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def get_all_for_export() -> list:
    """
    CSV/JSON export için tüm kayıtları tam olarak döner (b64 hariç — boyut yönetimi).
    """
    conn = get_connection()
    rows = conn.execute('''
        SELECT id, timestamp, filename, prediction, confidence,
               warning, warning_msg, probabilities, region_scores, mode, batch_id
        FROM predictions
        ORDER BY id DESC
    ''').fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d['probabilities'] = json.loads(d.get('probabilities') or '{}')
        d['region_scores']  = json.loads(d.get('region_scores') or '{}')
        results.append(d)
    return results
