from pydantic import BaseModel, Field

class TelemetryInput(BaseModel):
    volt: float = Field(..., description="Tegangan listrik sensor saat ini", example=170.5)
    rotate: float = Field(..., description="Kecepatan rotasi mesin (RPM)", example=440.2)
    pressure: float = Field(..., description="Tekanan mesin", example=100.8)
    vibration: float = Field(..., description="Tingkat getaran mesin", example=40.3)
    age: int = Field(..., description="Usia mesin dalam tahun", example=18)
    volt_mean_24h: float = Field(..., description="Rata-rata tegangan 24 jam terakhir", example=171.0)
    volt_std_24h: float = Field(..., description="Fluktuasi tegangan 24 jam terakhir", example=12.5)
    rotate_mean_24h: float = Field(..., description="Rata-rata rotasi 24 jam terakhir", example=445.0)
    rotate_std_24h: float = Field(..., description="Fluktuasi rotasi 24 jam terakhir", example=45.1)
    pressure_mean_24h: float = Field(..., description="Rata-rata tekanan 24 jam terakhir", example=100.2)
    pressure_std_24h: float = Field(..., description="Fluktuasi tekanan 24 jam terakhir", example=10.1)
    vibration_mean_24h: float = Field(..., description="Rata-rata getaran 24 jam terakhir", example=40.1)
    vibration_std_24h: float = Field(..., description="Fluktuasi getaran 24 jam terakhir", example=5.2)

class PredictionOutput(BaseModel):
    status: str = Field(..., example="NORMAL")
    failure_probability: float = Field(..., example=0.0412)
    recommendation: str = Field(..., example="Mesin beroperasi dalam batas aman.")