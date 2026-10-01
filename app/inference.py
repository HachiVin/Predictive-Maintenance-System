import joblib
import pandas as pd
import os

class ModelInference:
    def __init__(self, model_path: str = "models/rf_telemetry_v1.pkl"):
        if not os.path.exists(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, model_path)
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run notebook 02 first.")
            
        self.model = joblib.load(model_path)

    def predict(self, input_data: dict) -> dict:
        df_input = pd.DataFrame([input_data])

        if hasattr(self.model, "feature_names_in_"):
            expected_cols = list(self.model.feature_names_in_)
            for col in expected_cols:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[expected_cols]

        # Menghitung probabilitas dan hasil prediksi
        prob = float(self.model.predict_proba(df_input)[0][1])

        # Logika 
        if prob > 0.5:
            status = "CRITICAL_WARNING"
            recommendation = "PERINGATAN: Kerusakan komponen terdeteksi! Lakukan inspeksi teknisi dalam 24 jam."
        elif prob > 0.3:
            status = "WARNING"
            recommendation = "Anomali getaran/suhu meningkat. Jadwalkan pemeliharaan preventif."
        else:
            status = "NORMAL"
            recommendation = "Mesin beroperasi dalam batas kondisi aman."

        return {
            "status": status,
            "failure_probability": round(prob, 4),
            "recommendation": recommendation
        }