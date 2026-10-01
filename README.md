# Sistem Predictive Maintenance

Pipeline *machine learning end-to-end* untuk pemantauan peralatan industri secara *real-time*. Sistem ini mendeteksi anomali dan memprediksi potensi kerusakan mesin sebelum menyebabkan *unplanned downtime*.

Repositori ini berisi seluruh alur kerja: *preprocessing* data, pelatihan model, *backend* API untuk inferensi, dan *dashboard* Streamlit untuk pemantauan armada mesin secara *live*.

## Konteks & Nilai Bisnis
Di industri manufaktur, pemeliharaan reaktif (menunggu mesin rusak) memicu biaya *downtime* yang sangat mahal. Proyek ini mengimplementasikan pendekatan proaktif berbasis kondisi data (*condition-based maintenance*).
- Mengkonsumsi data telemetri mentah (voltase, rotasi, tekanan, getaran).
- Menggunakan *rolling window* 24 jam untuk menangkap pola degradasi temporal.
- Menghasilkan log peringatan yang dapat dieksekusi (*actionable alerts*) oleh tim teknisi.

## Sumber Data & Konteks Simulasi
Sistem ini dilatih dan divalidasi menggunakan **Microsoft Azure Predictive Maintenance Dataset**, sebuah *benchmark* standar industri yang berisi rekaman historis operasional dari 100 mesin berat di lapangan.
- **Fitur Telemetri:** 4 sensor fisik utama yaitu Tegangan Listrik (*Voltage*), Putaran Mesin (*Rotation*), Tekanan (*Pressure*), dan Getaran (*Vibration*).
- **Pola Kegagalan:** Model dilatih untuk mendeteksi fluktuasi anomali pada sensor dalam *rolling window* 24 jam sebelum komponen mesin benar-benar rusak.

**Konteks Simulasi UI:** Pada operasional pabrik sungguhan, data telemetri ini dikirimkan langsung oleh alat *Sensor Node IoT* di tiap mesin ke server terpusat via *message broker* (misal: Apache Kafka). Pada repositori ini, *dashboard* Streamlit bertindak sebagai simulator IoT yang secara sekuensial "menembakkan" data historis ke *endpoint* API, menciptakan pengalaman pemantauan CCTV *real-time* yang identik dengan layar kontrol pabrik.

## Analisis Cost-Benefit (Implementasi Dunia Nyata)
Pemasangan alat IoT pada mesin lama (*legacy equipment*) sangat efektif secara biaya jika menggunakan strategi **Asset Criticality**. Di lapangan, sistem AI ini difokuskan pada **Mesin Kritis (Tier 1)** yang menjadi urat nadi produksi. Biaya investasi pengadaan *Sensor Node* sangat terjangkau dan akan langsung *break-even* (balik modal) hanya dengan mencegah 1 jam *unplanned downtime* yang kerugiannya bisa mencapai puluhan hingga ratusan juta rupiah per jam.

## Tech Stack
- **Pemodelan:** Scikit-Learn (Random Forest), Pandas, NumPy
- **API / Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Streamlit
- **Deployment:** Docker & Streamlit Community Cloud (Standalone Edition)

## 🚀 Live Demo & Arsitektur Deployment Baru
Untuk keperluan demonstrasi publik, versi *live demo* di-*deploy* di **Streamlit Community Cloud** menggunakan pendekatan **Standalone**. Model AI dimuat langsung ke memori (`@st.cache_resource`) alih-alih melakukan HTTP Request ke API terpisah. Hal ini memastikan *zero network latency*, menekan biaya server, dan menghindari risiko pembatasan kuota *cloud provider*.
👉 **[KLIK DI SINI UNTUK MELIHAT LIVE DEMO DASHBOARD](MASUKKAN_URL_STREAMLIT_ANDA_DI_SINI)**

## Cara Menjalankan (Quick Start)

API *backend* telah dikontainerisasi menggunakan Docker untuk memastikan konsistensi *environment*. UI *frontend* berjalan secara lokal dan terhubung langsung ke API.

### 1. Jalankan API (Docker)
Pastikan Docker sudah berjalan di sistem Anda, lalu *build* dan jalankan *container*:

```bash
docker build -t pred-maint-api .
docker run -d -p 8000:8000 --name api-mesin pred-maint-api
```
API akan berjalan di `http://localhost:8000`. Dokumentasi interaktif (Swagger UI) dapat diakses melalui `http://localhost:8000/docs`.

### 2. Jalankan Dashboard
Buka terminal terpisah, instal dependensi UI, lalu jalankan aplikasi Streamlit:

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```
*Dashboard* akan otomatis terbuka di *browser* Anda pada alamat `http://localhost:8501`.

## Keputusan Arsitektur & Trade-off
- **Random Forest vs. Neural Networks (LSTM):** Meskipun LSTM dirancang untuk data *time-series*, Random Forest dipilih untuk iterasi ini. Alasannya: waktu *training* yang lebih cepat, kebutuhan komputasi inferensi yang lebih ringan, dan *explainability* (*feature importance*) yang tinggi—hal krusial saat menjelaskan hasil ke tim operasional lapangan.
- **Optimasi High Recall:** Dataset sangat tidak seimbang (*highly imbalanced* karena kerusakan jarang terjadi). Model menggunakan `class_weight='balanced'` dan memprioritaskan metrik *Recall* di atas *Precision*. Di skenario pabrik, *false alarm* (memeriksa mesin sehat) jauh lebih murah biayanya daripada melewatkan kerusakan kritis.
- **Stateless API:** *Service* FastAPI ini bersifat *stateless* dan mengekspektasikan fitur agregasi (*rolling window* 24 jam) sudah dikalkulasi di dalam *payload*. Pada skala *production* sesungguhnya, agregasi temporal ini akan ditangani terlebih dahulu oleh *streaming engine* (seperti Apache Kafka) sebelum masuk ke *endpoint* API.

## Struktur Payload API
**Endpoint:** `POST /predict`

```json
{
  "volt": 170.5,
  "rotate": 450.2,
  "pressure": 98.6,
  "vibration": 42.1,
  "age": 18,
  "volt_mean_24h": 171.0,
  "volt_std_24h": 12.5,
  "rotate_mean_24h": 445.0,
  "rotate_std_24h": 45.1,
  "pressure_mean_24h": 100.2,
  "pressure_std_24h": 10.1,
  "vibration_mean_24h": 40.1,
  "vibration_std_24h": 5.2
}
```