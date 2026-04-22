# services/report_service.py
# CSV ve JSON rapor üretme servisi.
# Batch tahminlerin sonuçlarını dışa aktarmak için kullanılır.

import csv
import json
import os
from datetime import datetime


class ReportService:
    """
    Tahmin sonuçlarını CSV veya JSON olarak dışa aktaran servis.

    Kullanım:
        service = ReportService()
        service.add_result(result_dict)
        service.export_csv('outputs/rapor.csv')
        service.export_json('outputs/rapor.json')
    """

    def __init__(self):
        self.results: list[dict] = []

    def add_result(self, result: dict):
        """
        Bir tahmin sonucunu listeye ekler.

        Args:
            result: InferenceService.predict() çıktısı
                    Opsiyonel olarak 'filename', 'true_label' eklenebilir.
        """
        entry = {
            'timestamp'     : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename'      : result.get('filename', ''),
            'prediction'    : result.get('prediction', ''),
            'confidence'    : result.get('confidence', 0.0),
            'warning'       : result.get('warning', False),
            'true_label'    : result.get('true_label', ''),
            **{
                f'prob_{cls}': result.get('probabilities', {}).get(cls, 0.0)
                for cls in ['3AA', '3CC', '4AA', 'Cleavage']
            }
        }
        self.results.append(entry)

    def add_batch(self, results: list[dict]):
        """Birden fazla sonucu ekler."""
        for r in results:
            self.add_result(r)

    def export_csv(self, output_path: str) -> str:
        """
        Sonuçları CSV dosyasına yazar.

        Args:
            output_path: Çıkış dosya yolu

        Returns:
            output_path
        """
        if not self.results:
            raise ValueError('Dışa aktarılacak sonuç yok.')

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fieldnames = list(self.results[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f'✅ CSV raporu kaydedildi: {output_path}')
        return output_path

    def export_json(self, output_path: str) -> str:
        """
        Sonuçları JSON dosyasına yazar.

        Args:
            output_path: Çıkış dosya yolu

        Returns:
            output_path
        """
        if not self.results:
            raise ValueError('Dışa aktarılacak sonuç yok.')

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(self.results),
            'results'      : self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f'✅ JSON raporu kaydedildi: {output_path}')
        return output_path

    def get_summary(self) -> dict:
        """
        Tahmin özetini döner (toplam, sınıf dağılımı, uyarı sayısı).
        """
        if not self.results:
            return {}

        from collections import Counter
        pred_counts = Counter(r['prediction'] for r in self.results)
        warning_count = sum(1 for r in self.results if r['warning'])

        return {
            'total'        : len(self.results),
            'distribution' : dict(pred_counts),
            'warning_count': warning_count,
            'avg_confidence': round(
                sum(r['confidence'] for r in self.results) / len(self.results), 4
            )
        }

    def clear(self):
        """Sonuç listesini temizler."""
        self.results = []
