"""
Aplikasi Prediksi Dropout Siswa - Jaya Jaya Institut
Prototype Machine Learning menggunakan Streamlit
Binary Classification: Dropout (1) vs Graduate (0)
Author: Failasuf Indi M
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🎓 Prediksi Dropout - Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #4a90d9 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0;
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Prediction result cards */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    .result-dropout {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    .result-graduate {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
        color: white;
    }
    .result-card h2 {
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
    }
    .result-card p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e8ecf1;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .metric-card h3 {
        color: #1e3a5f;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 700;
    }
    .metric-card p {
        color: #6b7c93;
        font-size: 0.85rem;
        margin: 0.3rem 0 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fc;
    }

    /* Section headers */
    .section-header {
        color: #1e3a5f;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8ecf1;
    }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8efff 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4a90d9;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #2c3e50;
    }

    /* Probability bar */
    .prob-container {
        background: #f0f2f5;
        border-radius: 20px;
        padding: 4px;
        margin: 0.5rem 0;
    }
    .prob-bar {
        height: 24px;
        border-radius: 16px;
        transition: width 0.8s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #d1d8e0, transparent);
        margin: 2rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #a0aec0;
        font-size: 0.85rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model')

    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))

    eval_path = os.path.join(model_dir, 'evaluation_results.json')
    with open(eval_path, 'r') as f:
        eval_results = json.load(f)

    feat_imp_path = os.path.join(model_dir, 'feature_importance.csv')
    feat_imp = None
    if os.path.exists(feat_imp_path):
        feat_imp = pd.read_csv(feat_imp_path)

    return model, scaler, feature_names, eval_results, feat_imp

try:
    model, scaler, feature_names, eval_results, feat_imp = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model belum tersedia. Jalankan `train_pipeline.py` terlebih dahulu.\n\nError: {e}")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎯 Navigasi")
    page = st.radio(
        "Pilih halaman:",
        ["🔮 Prediksi Individual", "📂 Prediksi Batch (CSV)", "📊 Model Performance", "📋 Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <p style='color: #6b7c93; font-size: 0.8rem;'>
            Dibuat oleh<br>
            <strong>Failasuf Indi M</strong><br>
            Dicoding ID: failasuf
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# COURSE MAPPING
# ============================================================
COURSE_MAP = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening)"
}

MARITAL_MAP = {
    1: "Single", 2: "Married", 3: "Widower",
    4: "Divorced", 5: "Facto Union", 6: "Legally Separated"
}

# ============================================================
# PAGE: PREDIKSI
# ============================================================
if page == "🔮 Prediksi Individual" and model_loaded:
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Sistem Prediksi Dropout Siswa</h1>
        <p>Jaya Jaya Institut — Early Warning System untuk Deteksi Risiko Dropout</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 <strong>Cara Penggunaan:</strong> Masukkan data profil siswa pada form di bawah, lalu klik tombol <strong>"🔍 Prediksi Sekarang"</strong> untuk melihat prediksi apakah siswa berisiko <strong>Dropout</strong> atau berpotensi <strong>Graduate</strong>.
    </div>
    """, unsafe_allow_html=True)

    # Input form
    with st.form("prediction_form"):
        st.markdown('<p class="section-header">📝 Data Profil Siswa</p>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 Personal", "🎓 Akademik Masuk", "📚 Performa Semester", "💰 Finansial & Lainnya"
        ])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                marital = st.selectbox("Status Pernikahan",
                    options=list(MARITAL_MAP.keys()),
                    format_func=lambda x: MARITAL_MAP[x],
                    index=0
                )
                gender = st.selectbox("Gender", options=[0, 1],
                    format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki"
                )
            with col2:
                age = st.number_input("Usia Saat Mendaftar", min_value=17, max_value=70, value=20)
                nationality = st.number_input("Kode Nasionalitas", min_value=1, max_value=109, value=1,
                    help="1=Portuguese, 2=German, 41=Brazilian, dll.")
            with col3:
                displaced = st.selectbox("Displaced", options=[0, 1],
                    format_func=lambda x: "Tidak" if x == 0 else "Ya"
                )
                international = st.selectbox("Mahasiswa Internasional", options=[0, 1],
                    format_func=lambda x: "Tidak" if x == 0 else "Ya"
                )

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                course = st.selectbox("Program Studi",
                    options=list(COURSE_MAP.keys()),
                    format_func=lambda x: COURSE_MAP[x]
                )
                app_mode = st.number_input("Application Mode", min_value=1, max_value=57, value=1,
                    help="1=1st phase general, 17=2nd phase, dll.")
                app_order = st.slider("Application Order", min_value=0, max_value=9, value=1,
                    help="0=pilihan pertama, 9=pilihan terakhir")
                daytime = st.selectbox("Waktu Kuliah", options=[0, 1],
                    format_func=lambda x: "Malam" if x == 0 else "Siang"
                )
            with col2:
                prev_qual = st.number_input("Previous Qualification", min_value=1, max_value=43, value=1,
                    help="1=Secondary Education, dll.")
                prev_qual_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=130.0, step=0.1)
                admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=130.0, step=0.1)
                mothers_qual = st.number_input("Mother's Qualification", min_value=1, max_value=44, value=1)

            col3, col4 = st.columns(2)
            with col3:
                fathers_qual = st.number_input("Father's Qualification", min_value=1, max_value=44, value=1)
                mothers_occ = st.number_input("Mother's Occupation", min_value=0, max_value=194, value=0)
            with col4:
                fathers_occ = st.number_input("Father's Occupation", min_value=0, max_value=194, value=0)
                edu_special = st.selectbox("Educational Special Needs", options=[0, 1],
                    format_func=lambda x: "Tidak" if x == 0 else "Ya"
                )

        with tab3:
            st.markdown("**Semester 1**")
            col1, col2, col3 = st.columns(3)
            with col1:
                sem1_credited = st.number_input("Units Credited (Sem 1)", min_value=0, max_value=30, value=0)
                sem1_enrolled = st.number_input("Units Enrolled (Sem 1)", min_value=0, max_value=30, value=6)
            with col2:
                sem1_evaluations = st.number_input("Units Evaluations (Sem 1)", min_value=0, max_value=50, value=6)
                sem1_approved = st.number_input("Units Approved (Sem 1)", min_value=0, max_value=30, value=5)
            with col3:
                sem1_grade = st.number_input("Grade Rata-rata (Sem 1)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
                sem1_without = st.number_input("Units Without Eval (Sem 1)", min_value=0, max_value=20, value=0)

            st.markdown("---")
            st.markdown("**Semester 2**")
            col4, col5, col6 = st.columns(3)
            with col4:
                sem2_credited = st.number_input("Units Credited (Sem 2)", min_value=0, max_value=30, value=0)
                sem2_enrolled = st.number_input("Units Enrolled (Sem 2)", min_value=0, max_value=30, value=6)
            with col5:
                sem2_evaluations = st.number_input("Units Evaluations (Sem 2)", min_value=0, max_value=50, value=6)
                sem2_approved = st.number_input("Units Approved (Sem 2)", min_value=0, max_value=30, value=5)
            with col6:
                sem2_grade = st.number_input("Grade Rata-rata (Sem 2)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
                sem2_without = st.number_input("Units Without Eval (Sem 2)", min_value=0, max_value=20, value=0)

        with tab4:
            col1, col2, col3 = st.columns(3)
            with col1:
                tuition = st.selectbox("Tuition Fees Up to Date", options=[0, 1],
                    format_func=lambda x: "Tidak" if x == 0 else "Ya",
                    index=1
                )
                debtor = st.selectbox("Status Debtor", options=[0, 1],
                    format_func=lambda x: "Tidak" if x == 0 else "Ya"
                )
            with col2:
                scholarship = st.selectbox("Penerima Beasiswa", options=[0, 1],
                    format_func=lambda x: "Tidak" if x == 0 else "Ya"
                )
                unemployment = st.number_input("Unemployment Rate (%)", min_value=0.0, max_value=30.0, value=10.8, step=0.1)
            with col3:
                inflation = st.number_input("Inflation Rate (%)", min_value=-5.0, max_value=10.0, value=1.4, step=0.1)
                gdp = st.number_input("GDP", min_value=-10.0, max_value=10.0, value=1.74, step=0.01)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "🔍 Prediksi Sekarang",
            use_container_width=True,
            type="primary"
        )

    # Process prediction
    if submitted:
        # Build feature array
        sem1_approval_rate = sem1_approved / sem1_enrolled if sem1_enrolled > 0 else 0
        sem2_approval_rate = sem2_approved / sem2_enrolled if sem2_enrolled > 0 else 0
        total_approved = sem1_approved + sem2_approved
        avg_grade = (sem1_grade + sem2_grade) / 2

        input_data = pd.DataFrame([{
            'Marital_status': marital,
            'Application_mode': app_mode,
            'Application_order': app_order,
            'Course': course,
            'Daytime_evening_attendance': daytime,
            'Previous_qualification': prev_qual,
            'Previous_qualification_grade': prev_qual_grade,
            'Nacionality': nationality,
            'Mothers_qualification': mothers_qual,
            'Fathers_qualification': fathers_qual,
            'Mothers_occupation': mothers_occ,
            'Fathers_occupation': fathers_occ,
            'Admission_grade': admission_grade,
            'Displaced': displaced,
            'Educational_special_needs': edu_special,
            'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition,
            'Gender': gender,
            'Scholarship_holder': scholarship,
            'Age_at_enrollment': age,
            'International': international,
            'Curricular_units_1st_sem_credited': sem1_credited,
            'Curricular_units_1st_sem_enrolled': sem1_enrolled,
            'Curricular_units_1st_sem_evaluations': sem1_evaluations,
            'Curricular_units_1st_sem_approved': sem1_approved,
            'Curricular_units_1st_sem_grade': sem1_grade,
            'Curricular_units_1st_sem_without_evaluations': sem1_without,
            'Curricular_units_2nd_sem_credited': sem2_credited,
            'Curricular_units_2nd_sem_enrolled': sem2_enrolled,
            'Curricular_units_2nd_sem_evaluations': sem2_evaluations,
            'Curricular_units_2nd_sem_approved': sem2_approved,
            'Curricular_units_2nd_sem_grade': sem2_grade,
            'Curricular_units_2nd_sem_without_evaluations': sem2_without,
            'Unemployment_rate': unemployment,
            'Inflation_rate': inflation,
            'GDP': gdp,
            'Sem1_approval_rate': sem1_approval_rate,
            'Sem2_approval_rate': sem2_approval_rate,
            'Total_approved': total_approved,
            'Avg_grade': avg_grade,
        }])

        # Ensure column order matches training
        input_data = input_data[feature_names]

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        # prediction: 1 = Dropout, 0 = Graduate
        predicted_label = "Dropout" if prediction == 1 else "Graduate"
        prob_dropout = probabilities[1]   # kelas 1 = Dropout
        prob_graduate = probabilities[0]  # kelas 0 = Graduate

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">📊 Hasil Prediksi</p>', unsafe_allow_html=True)

        # Result card
        if predicted_label == "Dropout":
            card_class = "result-dropout"
            emoji = "🚨"
            message = "Siswa ini memiliki risiko TINGGI untuk dropout!"
        else:
            card_class = "result-graduate"
            emoji = "🎉"
            message = "Siswa ini diprediksi akan berhasil lulus!"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <h2>{emoji} {predicted_label}</h2>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability display (2 kelas: Dropout & Graduate)
        st.markdown("#### Distribusi Probabilitas")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #ff6b6b">{prob_dropout*100:.1f}%</h3>
                <p>🚨 Dropout</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob_dropout))

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #26de81">{prob_graduate*100:.1f}%</h3>
                <p>🎓 Graduate</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob_graduate))

        # Risk factors explanation
        if predicted_label == "Dropout":
            st.markdown("---")
            st.warning("⚠️ **Rekomendasi Tindakan:**")
            st.markdown("""
            - 🎯 **Bimbingan akademik intensif** — Hubungi siswa untuk sesi konseling
            - 💰 **Evaluasi bantuan finansial** — Periksa status beasiswa dan tunggakan
            - 📊 **Monitoring berkala** — Jadwalkan evaluasi performa mingguan
            - 🤝 **Program mentoring** — Pasangkan dengan mahasiswa senior sebagai mentor
            """)
        else:
            st.success("✅ **Siswa ini diprediksi pada jalur yang baik.** Tetap lakukan monitoring rutin untuk memastikan performa akademik tetap terjaga.")

# ============================================================
# PAGE: PREDIKSI BATCH
# ============================================================
elif page == "📂 Prediksi Batch (CSV)" and model_loaded:
    st.markdown("""
    <div class="main-header">
        <h1>📂 Prediksi Batch (Siswa Enrolled)</h1>
        <p>Gunakan fitur ini untuk memprediksi risiko dropout dari sekumpulan data siswa (contoh: siswa yang masih Enrolled)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 <strong>Cara Penggunaan:</strong> Upload file CSV yang berisi data siswa (misalnya <code>enrolled_data.csv</code>). 
        Sistem akan memproses semua baris sekaligus dan memberikan daftar siswa mana saja yang memiliki probabilitas tinggi untuk Dropout.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload file CSV (Gunakan enrolled_data.csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"File berhasil di-upload! Memproses {len(df_batch)} baris data...")

            # Pastikan fitur engineered sudah ada, jika belum buatkan
            if 'Sem1_approval_rate' not in df_batch.columns:
                df_batch['Sem1_approval_rate'] = np.where(
                    df_batch['Curricular_units_1st_sem_enrolled'] > 0,
                    df_batch['Curricular_units_1st_sem_approved'] / df_batch['Curricular_units_1st_sem_enrolled'],
                    0
                )
            if 'Sem2_approval_rate' not in df_batch.columns:
                df_batch['Sem2_approval_rate'] = np.where(
                    df_batch['Curricular_units_2nd_sem_enrolled'] > 0,
                    df_batch['Curricular_units_2nd_sem_approved'] / df_batch['Curricular_units_2nd_sem_enrolled'],
                    0
                )
            if 'Total_approved' not in df_batch.columns:
                df_batch['Total_approved'] = df_batch['Curricular_units_1st_sem_approved'] + df_batch['Curricular_units_2nd_sem_approved']
            if 'Avg_grade' not in df_batch.columns:
                df_batch['Avg_grade'] = (df_batch['Curricular_units_1st_sem_grade'] + df_batch['Curricular_units_2nd_sem_grade']) / 2

            # Ekstrak fitur sesuai model
            X_batch = df_batch[feature_names]

            # Prediksi
            X_scaled = scaler.transform(X_batch)
            predictions = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1] # Ambil probabilitas untuk kelas Dropout (1)

            # Tambahkan hasil ke dataframe
            df_result = df_batch.copy()
            df_result['Prediction'] = ["Dropout" if p == 1 else "Graduate" for p in predictions]
            df_result['Dropout_Probability'] = probs

            # Tampilkan Ringkasan
            total = len(df_result)
            total_dropout = (df_result['Prediction'] == 'Dropout').sum()
            total_graduate = total - total_dropout

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('### 📊 Ringkasan Prediksi Batch')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{total}</h3><p>Total Siswa Diproses</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{total_dropout}</h3><p style="color: #ff6b6b;">Berisiko Dropout</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>{total_graduate}</h3><p style="color: #26de81;">Potensi Lulus</p></div>', unsafe_allow_html=True)

            st.markdown("---")

            # Tampilkan Tabel Siswa Berisiko Tinggi
            st.markdown('### 🚨 Daftar Siswa Berisiko Dropout Tinggi')
            df_risk = df_result[df_result['Prediction'] == 'Dropout'].sort_values(by='Dropout_Probability', ascending=False)
            
            if len(df_risk) > 0:
                # Tampilkan kolom penting saja di tabel preview
                display_cols = ['Course', 'Age_at_enrollment', 'Tuition_fees_up_to_date', 
                              'Sem1_approval_rate', 'Sem2_approval_rate', 'Dropout_Probability']
                
                # Format tabel agar lebih mudah dibaca
                df_display = df_risk[display_cols].copy()
                df_display['Dropout_Probability'] = df_display['Dropout_Probability'].apply(lambda x: f"{x*100:.1f}%")
                df_display['Course'] = df_display['Course'].map(COURSE_MAP).fillna(df_display['Course'])
                
                st.dataframe(df_display, use_container_width=True)

                # Tombol Download Hasil
                csv_data = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Hasil Prediksi Lengkap (CSV)",
                    data=csv_data,
                    file_name='hasil_prediksi_batch.csv',
                    mime='text/csv',
                    type="primary"
                )
            else:
                st.success("🎉 Tidak ada siswa yang terdeteksi berisiko tinggi untuk Dropout dalam data ini.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.info("Pastikan file CSV memiliki kolom yang sama dengan dataset training awal.")

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "📊 Model Performance" and model_loaded:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Performa Model</h1>
        <p>Evaluasi dan metrik model Machine Learning — Binary Classification: Dropout vs Graduate</p>
    </div>
    """, unsafe_allow_html=True)

    # Model info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{eval_results.get('accuracy', 0)*100:.1f}%</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{eval_results.get('f1_weighted', 0)*100:.1f}%</h3>
            <p>F1-Score (Weighted)</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        roc = eval_results.get('roc_auc', 0) or 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{roc*100:.1f}%</h3>
            <p>ROC-AUC</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        recall_do = eval_results.get('recall_dropout', 0) or 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{recall_do*100:.1f}%</h3>
            <p>Recall Dropout</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Data info
    data_info = eval_results.get('data_info', {})
    if data_info:
        st.markdown('<p class="section-header">📊 Info Dataset</p>', unsafe_allow_html=True)
        ci1, ci2, ci3, ci4 = st.columns(4)
        with ci1:
            st.metric("Total Siswa (Full)", f"{data_info.get('total_students', 0):,}")
        with ci2:
            st.metric("Siswa Enrolled (Dipisahkan)", f"{data_info.get('enrolled_students', 0):,}")
        with ci3:
            st.metric("Dropout (Training)", f"{data_info.get('dropout_count', 0):,}")
        with ci4:
            st.metric("Graduate (Training)", f"{data_info.get('graduate_count', 0):,}")
        st.info("ℹ️ Siswa berstatus **Enrolled** dipisahkan dari proses training karena belum memiliki label akhir. Data ini disimpan di `enrolled_data.csv` untuk prediksi masa depan.")

    # Classification report
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<p class="section-header">📋 Classification Report</p>', unsafe_allow_html=True)
        report = eval_results.get('classification_report', {})
        report_data = []
        for cls in ['Graduate', 'Dropout']:
            if cls in report:
                r = report[cls]
                report_data.append({
                    'Class': cls,
                    'Precision': f"{r['precision']:.4f}",
                    'Recall': f"{r['recall']:.4f}",
                    'F1-Score': f"{r['f1-score']:.4f}",
                    'Support': int(r['support'])
                })
        if report_data:
            st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

        # Model comparison
        st.markdown('<p class="section-header">⚔️ Perbandingan Model</p>', unsafe_allow_html=True)
        all_results = eval_results.get('all_model_results', {})
        if all_results:
            compare_data = []
            for name, r in all_results.items():
                compare_data.append({
                    'Model': name,
                    'Accuracy': f"{r['accuracy']:.4f}",
                    'F1 (Weighted)': f"{r['f1_weighted']:.4f}",
                    'Recall Dropout': f"{r['recall_dropout']:.4f}",
                })
            st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<p class="section-header">🔑 Top Feature Importance</p>', unsafe_allow_html=True)
        if feat_imp is not None:
            top_features = feat_imp.head(15)
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]
            bars = ax.barh(
                range(len(top_features)),
                top_features['importance'].values[::-1],
                color=colors_gradient
            )
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'].values[::-1], fontsize=9)
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title('Top 15 Most Important Features', fontsize=13, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Feature importance data tidak tersedia.")

        # Confusion matrix
        st.markdown('<p class="section-header">🎯 Confusion Matrix</p>', unsafe_allow_html=True)
        cm = eval_results.get('confusion_matrix', [])
        if cm:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Graduate', 'Dropout'],
                yticklabels=['Graduate', 'Dropout'],
                ax=ax, linewidths=0.5, linecolor='white'
            )
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.set_title('Confusion Matrix (Binary)', fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

    # Best params
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    with st.expander("🔧 Best Hyperparameters"):
        st.json(eval_results.get('best_params', {}))

# ============================================================
# PAGE: TENTANG
# ============================================================
elif page == "📋 Tentang":
    st.markdown("""
    <div class="main-header">
        <h1>📋 Tentang Proyek</h1>
        <p>Informasi tentang proyek prediksi dropout Jaya Jaya Institut</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🏫 Latar Belakang

    **Jaya Jaya Institut** merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000.
    Hingga saat ini telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak
    juga siswa yang tidak menyelesaikan pendidikannya alias **dropout**.

    Jumlah dropout yang tinggi menjadi masalah besar bagi institusi. Oleh karena itu, dibangun sistem prediksi
    berbasis **Machine Learning** untuk mendeteksi siswa yang berpotensi dropout sedini mungkin.

    ### 🎯 Tujuan
    - Memprediksi apakah seorang siswa akan **Dropout** atau **Graduate** (binary classification)
    - Memberikan *early warning* agar siswa berisiko dapat diberikan bimbingan khusus
    - Membantu institusi dalam mengambil keputusan berbasis data

    ### 📊 Dataset & Pendekatan
    Dataset berisi data siswa dengan **36 fitur** yang mencakup:
    - Informasi demografis (usia, gender, status pernikahan)
    - Latar belakang keluarga (pendidikan & pekerjaan orang tua)
    - Data akademik (nilai masuk, performa semester 1 & 2)
    - Status finansial (beasiswa, tunggakan, status pembayaran)
    - Kondisi makroekonomi (unemployment, inflation, GDP)

    **Catatan penting:** Hanya siswa berstatus **Dropout** dan **Graduate** yang digunakan untuk training model.
    Siswa berstatus **Enrolled** (belum memiliki label akhir) dipisahkan dan disimpan di `enrolled_data.csv`
    untuk keperluan prediksi masa depan.

    ### 🛠️ Teknologi
    | Komponen | Teknologi |
    |---|---|
    | Data Processing | Pandas, NumPy, Scikit-learn |
    | Visualization | Matplotlib, Seaborn |
    | Machine Learning | Random Forest / Gradient Boosting |
    | Handling Imbalance | SMOTE (imbalanced-learn) |
    | Prototype | Streamlit |
    | Dashboard | Google Looker Studio |

    ### 👤 Pengembang
    - **Nama:** Failasuf Indi M
    - **Email:** failasufindi123@gmail.com
    - **ID Dicoding:** failasuf
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2024 Jaya Jaya Institut — Student Dropout Prediction System</p>
    <p>Built with ❤️ using Streamlit | Failasuf Indi M</p>
</div>
""", unsafe_allow_html=True)
