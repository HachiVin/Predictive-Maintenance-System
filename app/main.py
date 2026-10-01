from fastapi import FastAPI, HTTPException
from app.schemas import TelemetryInput, PredictionOutput
from app.inference import ModelInference

app = FastAPI(
    title="Predictive Maintenance AI Engine",
    description="API Inferensi Real-Time untuk Pemantauan Kondisi Mesin Industri",
    version="1.0.0"
)

try:
    model_service = ModelInference()
except Exception as e:
    model_service = None
    print(f"Peringatan: Gagal memuat model (.pkl). Pastikan model sudah diekspor. Error: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Predictive Maintenance API",
        "model_loaded": model_service is not None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_telemetry(payload: TelemetryInput):
    if not model_service:
        raise HTTPException(
            status_code=500, 
            detail="Model AI belum dimuat. Jalankan notebook 02_model_experimentation.ipynb terlebih dahulu."
        )
    try:
        result = model_service.predict(payload.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")