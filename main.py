# main.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from datetime import datetime
import io, csv, json, uuid

from services.inference_service import InferenceService
from services.gradcam_service   import GradCAMService
from services.report_service    import ReportService

app = FastAPI(title="DeepEmbryo API")

# Servisler (model bir kez yüklenir)
inference = InferenceService('assets/best_model.pth')
gradcam   = GradCAMService('assets/best_model.pth')

# Batch session store (in-memory)
# { session_id: ReportService }
sessions: dict[str, ReportService] = {}


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "model": "EfficientNetB3"}


# ─────────────────────────────────────────────
# TEKLİ TAHMİN
# ─────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Tek embriyo görseli için tahmin yapar."""
    img = Image.open(io.BytesIO(await file.read()))
    return JSONResponse(inference.predict(img))


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────

@app.post("/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)):
    """Grad-CAM ısı haritası üretir. Base64 PNG döner."""
    img    = Image.open(io.BytesIO(await file.read()))
    result = gradcam.generate(img)
    return JSONResponse({
        "predicted_class": result["predicted_class"],
        "confidence"     : result["confidence"],
        "cam_image_b64"  : result["cam_image_b64"],
        "original_b64"   : result["original_b64"],
        "region_scores"  : result["region_scores"],
    })


# ─────────────────────────────────────────────
# BATCH TAHMİN
# ─────────────────────────────────────────────

@app.post("/batch")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Birden fazla görsel için toplu tahmin yapar.
    Sonuçlar session'da tutulur, export için session_id kullan.
    """
    report = ReportService()

    for file in files:
        img    = Image.open(io.BytesIO(await file.read()))
        result = inference.predict(img)
        result['filename'] = file.filename
        report.add_result(result)

    session_id = str(uuid.uuid4())
    sessions[session_id] = report

    return JSONResponse({
        "session_id": session_id,
        "summary"   : report.get_summary(),
        "results"   : report.results,
    })


# ─────────────────────────────────────────────
# EXPORT (CSV / JSON)
# ─────────────────────────────────────────────

@app.get("/export/{session_id}")
async def export(
    session_id: str,
    format: str = Query(default="csv", pattern="^(csv|json)$")
):
    """
    Batch tahmin sonuçlarını indirir.

    Parametreler:
        session_id : /batch endpoint'inden dönen ID
        format     : 'csv' veya 'json' (varsayılan: csv)

    Örnek:
        GET /export/abc-123?format=csv
        GET /export/abc-123?format=json
    """
    if session_id not in sessions:
        return JSONResponse(
            {"error": "Session bulunamadı. Önce /batch çağırın."},
            status_code=404
        )

    report    = sessions[session_id]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if format == "csv":
        output     = io.StringIO()
        fieldnames = list(report.results[0].keys())
        writer     = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report.results)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    f"attachment; filename=deepembryo_{timestamp}.csv"
            }
        )

    # format == "json"
    data = {
        "generated_at" : timestamp,
        "total_samples": len(report.results),
        "summary"      : report.get_summary(),
        "results"      : report.results,
    }
    output = io.StringIO(json.dumps(data, ensure_ascii=False, indent=2))

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="application/json",
        headers={
            "Content-Disposition":
                f"attachment; filename=deepembryo_{timestamp}.json"
        }
    )