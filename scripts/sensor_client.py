import time
import requests
import pandas as pd
import os

API_URL = "http://localhost:8000/predict"
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/features_ready.csv")

def start_simulation():
    print("=" * 60)
    print("🤖 SENSOR CLIENT SIMULATOR (PABRIK INTEGRASI)")
    print(f"Target API: {API_URL}")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"❌ File data tidak ditemukan di {DATA_PATH}.")
        print("Silakan jalankan notebook 01_eda_and_preprocessing.ipynb lebih dulu.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Kolom non-fitur yang dibuang sebelum dikirim ke API
    ignore_cols = ['machineID', 'datetime', 'error_type', 'failure_type', 'is_broken', 'model']
    feature_df = df.drop(columns=[c for c in ignore_cols if c in df.columns])

    print(f"📡 Mengirimkan aliran data sensor secara real-time ({len(feature_df)} baris data available)...")
    print("Tekan Ctrl+C untuk menghentikan simulasi.\n")

    for idx, row in feature_df.iterrows():
        payload = row.to_dict()
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                res = response.json()
                status = res['status']
                prob = res['failure_probability'] * 100
                rec = res['recommendation']

                if status == "CRITICAL_WARNING":
                    print(f"🔴 [CRITICAL] Baris #{idx} | Probabilitas Rusak: {prob:.1f}% | {rec}")
                elif status == "WARNING":
                    print(f"🟡 [WARNING]  Baris #{idx} | Probabilitas Rusak: {prob:.1f}% | {rec}")
                else:
                    print(f"🟢 [NORMAL]   Baris #{idx} | Probabilitas Rusak: {prob:.1f}%")
            else:
                print(f"❌ API Error ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            print("❌ Tidak dapat terhubung ke FastAPI server!")
            print("Pastikan server FastAPI sudah berjalan dengan perintah: uvicorn app.main:app --reload")
            break

        time.sleep(1.5) # Jeda pengiriman log setiap 1.5 detik

if __name__ == "__main__":
    start_simulation()