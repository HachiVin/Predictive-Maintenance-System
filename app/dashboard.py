import streamlit as st
import pandas as pd
import requests
import time

# Konfigurasi Halaman 
st.set_page_config(
    page_title="Predictive Maintenance System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Variabel Endpoint API Backend
API_URL = "http://127.0.0.1:8000/predict"

# Header Utama (Minimalis)
st.title("Predictive Maintenance System")
st.markdown("Real-time telemetri monitoring dan deteksi anomali mesin berbasis Machine Learning.")
st.markdown("---")

# Tab Layout
tab1, tab2, tab3 = st.tabs([
    "Overview", 
    "Live Monitoring", 
    "System Specifications"
])

# ==========================================
# TAB 1: OVERVIEW (Dengan Penjelasan Dataset)
# ==========================================
with tab1:
    st.subheader("Business Impact")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Objective**
        Sistem ini dirancang untuk meminimalkan *unplanned downtime* pada lantai produksi. Menggunakan algoritma klasifikasi untuk memproses aliran data sensor secara *real-time*, mendeteksi pola degradasi, dan memberikan peringatan dini sebelum kegagalan komponen terjadi.
        """)
    with col2:
        st.markdown("""
        **Key Metrics (Expected)**
        * Penurunan biaya pemeliharaan reaktif hingga 30%.
        * Reduksi *downtime* mesin hingga 45%.
        * Optimalisasi penjadwalan teknisi berbasis data aktual (*condition-based maintenance*).
        """)

    st.markdown("---")
    st.subheader("Data Source & Simulation Context")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        **Microsoft Azure Predictive Maintenance Dataset**
        Model AI ini dilatih menggunakan dataset *benchmark* standar industri dari Microsoft. Data ini merupakan rekaman historis operasional dari 100 mesin berat di lapangan.
        * **Fitur Utama:** 4 sensor fisik, yaitu Tegangan Listrik (*Voltage*), Putaran Mesin (*Rotation/RPM*), Tekanan (*Pressure*), dan Getaran (*Vibration*).
        * **Pola Kegagalan:** Algoritma dilatih untuk mengenali fluktuasi anomali pada sensor dalam jendela waktu 24 jam (*rolling window*) sebelum komponen benar-benar rusak.
        """)
    with col4:
        st.info("""
        💡 **Bagaimana Simulasi Dasbor Ini Bekerja?**
        Di pabrik nyata, data dikirim langsung oleh alat *Sensor Node IoT* ke server via jaringan (seperti Apache Kafka). 
        
        Untuk keperluan demonstrasi ini, sistem menggunakan aliran data historis yang 'ditembakkan' secara berurutan ke *endpoint* API AI. Ini menghasilkan simulasi *real-time monitoring* yang identik dengan layar kontrol di lantai pabrik sungguhan.
        """)

# ==========================================
# TAB 2: LIVE MONITORING (ENTERPRISE UI)
# ==========================================
with tab2:
    st.markdown("#### Fleet Status Dashboard")
    
    # Inisialisasi Memori
    if 'akumulasi_laporan' not in st.session_state:
        st.session_state.akumulasi_laporan = []
        
    # Kontrol Panel Minimalis
    col_btn1, col_btn2, col_space = st.columns([2, 3, 7])
    with col_btn1:
        mulai_inspeksi = st.button("Start Monitoring", use_container_width=True)
    with col_btn2:
        stop_inspeksi = st.button("Stop & Export Log", use_container_width=True)

    if stop_inspeksi:
        st.info("Sesi pemantauan dihentikan. Rekapitulasi log tersedia di bawah.")
        if len(st.session_state.akumulasi_laporan) > 0:
            df_akumulasi = pd.DataFrame(st.session_state.akumulasi_laporan)
            st.dataframe(df_akumulasi, use_container_width=True, hide_index=True)
            
            csv = df_akumulasi.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Log (CSV)",
                data=csv,
                file_name='maintenance_log.csv',
                mime='text/csv',
            )
        else:
            st.success("Status operasional optimal. Tidak ada log kerusakan pada sesi ini.")

    # Placeholder untuk UI dinamis
    dashboard_placeholder = st.empty()
    
    if mulai_inspeksi:
        st.session_state.akumulasi_laporan = []
        
        try:
            df = pd.read_csv('data/processed/features_ready.csv')
            df = df.fillna('None')
            
            while True:
                df_batch = df.sample(16).reset_index()
                hasil_semua = []
                
                for i, row in df_batch.iterrows():
                    mesin_id = f"MCH-{str(row['index']).zfill(4)}" 
                    payload = row.drop(['index', 'machineID'], errors='ignore').to_dict()
                    
                    try:
                        response = requests.post(API_URL, json=payload)
                        hasil = response.json()
                        
                        status = hasil.get('status', 'NORMAL')
                        prob = hasil.get('failure_probability', 0.0) * 100
                        
                        info_mesin = {
                            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "Equipment ID": mesin_id,
                            "Status": status,
                            "Probability": f"{prob:.1f}%",
                            "Action Required": hasil.get('recommendation', '-')
                        }
                        hasil_semua.append(info_mesin)
                        
                        if status in ["WARNING", "CRITICAL_WARNING"]:
                            st.session_state.akumulasi_laporan.append(info_mesin)
                            
                    except Exception:
                        pass 
                
                with dashboard_placeholder.container():
                    st.caption("Active connection to telemetry stream. Refreshing every 3 seconds...")
                    
                    # Menggambar Grid dengan gaya UI Enterprise (Minimalis)
                    for i in range(0, len(hasil_semua), 4):
                        cols = st.columns(4)
                        for j, col in enumerate(cols):
                            if i + j < len(hasil_semua):
                                data = hasil_semua[i + j]
                                
                                # Palet Warna Korporat (Bootstrap-like)
                                if data['Status'] == "CRITICAL_WARNING":
                                    warna = "#dc3545" # Merah elegan
                                    status_teks = "CRITICAL"
                                elif data['Status'] == "WARNING":
                                    warna = "#ffc107" # Kuning mustard
                                    status_teks = "WARNING"
                                else:
                                    warna = "#28a745" # Hijau kalem
                                    status_teks = "NORMAL"
                                    
                                with col:
                                    # Desain Kartu: Transparan dengan garis batas kiri
                                    st.markdown(f"""
                                    <div style="border-left: 5px solid {warna}; padding: 12px 16px; border-radius: 4px; background-color: rgba(128, 128, 128, 0.05); margin-bottom: 12px; border-top: 1px solid rgba(128,128,128,0.1); border-right: 1px solid rgba(128,128,128,0.1); border-bottom: 1px solid rgba(128,128,128,0.1);">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <p style="margin: 0; font-size: 16px; font-weight: 600; color: #ececec;">{data['Equipment ID']}</p>
                                            <p style="margin: 0; font-size: 12px; font-weight: 700; color: {warna};">{status_teks}</p>
                                        </div>
                                        <p style="margin: 8px 0 0 0; font-size: 13px; color: #a0a0a0;">Failure Probability: <span style="color: #cccccc;">{data['Probability']}</span></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown(f"**Actionable Alerts Log ({len(st.session_state.akumulasi_laporan)} records)**")
                    
                    if len(st.session_state.akumulasi_laporan) > 0:
                        df_laporan = pd.DataFrame(st.session_state.akumulasi_laporan)[::-1]
                        st.dataframe(df_laporan.head(8), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No anomalies detected in the current session.")
                
                time.sleep(3) 

        except FileNotFoundError:
            st.error("System Error: Telemetry source file not found.")

# ==========================================
# TAB 3: SYSTEM SPECIFICATIONS
# ==========================================
with tab3:
    st.subheader("Architecture & Model Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**API Backend (FastAPI)**")
        st.code("""
{
  "volt": 170.5,
  "rotate": 450.2,
  "pressure": 98.6,
  "vibration": 42.1
}
        """, language="json")
    with col2:
        st.markdown("**Machine Learning Engine**")
        st.markdown("""
        * **Algorithm:** Random Forest Classifier
        * **Feature Engineering:** 24-hour Rolling Window (Mean, Std)
        * **Class Handling:** SMOTE / Balanced Class Weights
        * **Pipeline:** Scikit-Learn
        """)
