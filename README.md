# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

**Jaya Jaya Institut** merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias **dropout**.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin **mendeteksi secepat mungkin** siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis

1. **Tingginya angka siswa yang tidak menyelesaikan pendidikan (dropout)** — Kondisi ini merugikan reputasi institusi dan berdampak pada keberlanjutan operasional Jaya Jaya Institut.
2. **Tidak ada sistem deteksi dini** — Institusi belum memiliki mekanisme otomatis untuk mengidentifikasi siswa yang berisiko dropout sedini mungkin.
3. **Kurangnya pemahaman data** — Pihak manajemen membutuhkan dashboard untuk memahami faktor-faktor yang memengaruhi performa dan risiko dropout siswa.

### Cakupan Proyek

1. Analisis data eksploratif (EDA) untuk memahami faktor-faktor yang memengaruhi dropout.
2. Pembuatan model machine learning (**binary classification: Dropout vs Graduate**) untuk memprediksi potensi seorang siswa akan dropout atau lulus.
3. Pembuatan business dashboard menggunakan Google Looker Studio untuk monitoring performa siswa.
4. Pembuatan prototype sistem prediksi menggunakan Streamlit yang di-deploy ke Streamlit Community Cloud.
5. Rekomendasi action items berdasarkan hasil analisis.

### Persiapan

Sumber data: [Dataset Students Performance - Dicoding](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Dataset berisi **4.424 baris** dan **37 kolom** (36 fitur + 1 target), mencakup informasi demografis, latar belakang keluarga, data akademik, status finansial, dan kondisi makroekonomi.

Setup environment:

```bash
# Buat virtual environment (opsional)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

## Business Dashboard

Dashboard dibuat menggunakan **Google Looker Studio** untuk membantu Jaya Jaya Institut dalam memahami data dan memonitor performa siswa.

Dashboard menampilkan:
- **Ringkasan KPI**: Total siswa, persentase Dropout/Graduate/Enrolled
- **Distribusi Status Siswa**: Pie chart proporsi status
- **Dropout Rate per Program Studi**: Bar chart untuk identifikasi program dengan risiko tertinggi
- **Performa Akademik vs Status**: Perbandingan rata-rata grade dan approval rate
- **Faktor Finansial**: Analisis pengaruh beasiswa, tunggakan, dan status pembayaran
- **Filter Interaktif**: Dropdown untuk filter berdasarkan course, gender, dan status pernikahan

**Link Dashboard:** [Looker Studio Dashboard](https://datastudio.google.com/reporting/bdf9bb17-bbab-469e-b8ea-4b1812d28909)

## Menjalankan Sistem Machine Learning

Prototype sistem prediksi dropout dibangun menggunakan **Streamlit** dan telah di-deploy ke Streamlit Community Cloud.

### Cara Menjalankan Secara Lokal

```bash
# Pastikan sudah di direktori submission2
cd submission2

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

### Fitur Aplikasi
1. **🔮 Prediksi Dropout** — Input data profil siswa dan dapatkan prediksi apakah siswa berisiko **Dropout** atau berpotensi **Graduate**, beserta probabilitasnya.
2. **📊 Model Performance** — Lihat metrik evaluasi model, confusion matrix, dan feature importance.
3. **📋 Tentang** — Informasi tentang proyek dan teknologi yang digunakan.

**Link Prototype (Streamlit Cloud):** [Streamlit App](https://app-ml-app-jayajayainstitut-arjlpkdx3cykrdd2bkaysw.streamlit.app/)
**Link Repository GitHub:** [GitHub Repo](https://github.com/sendyMar/streamlit-ml-app-jayajayainstitut)

## Conclusion

### Temuan Utama

1. **Distribusi Data:** Dari total 4.424 siswa dalam dataset, 49.9% berstatus Graduate (2.209), 32.1% Dropout (1.421), dan 17.9% masih berstatus Enrolled (794). Siswa **Enrolled dipisahkan** dari proses pemodelan karena belum memiliki label akhir — data ini disimpan di `enrolled_data.csv` untuk keperluan prediksi masa depan.

2. **Faktor Kunci Dropout (dari EDA):**
   - **Performa Akademik (paling dominan):** Siswa dropout rata-rata hanya menyelesaikan ~2.5 mata kuliah per semester dibanding ~6.2 untuk yang lulus. Nilai rata-rata semester 1 siswa dropout (7.26) jauh lebih rendah dari Graduate (12.64).
   - **Faktor Finansial:** Hanya 68% siswa dropout yang SPP-nya terbayar lunas (vs 99% Graduate), hanya 9% mendapat beasiswa (vs 38% Graduate), dan 22% berstatus debtor (vs 5% Graduate).
   - **Usia:** Rata-rata usia dropout (26.1 tahun) lebih tinggi dibanding yang lulus (21.8 tahun).
   - **Program Studi:** Biofuel Production Technologies (66.7%), Equinculture (55.3%), dan Informatics Engineering (54.1%) memiliki dropout rate tertinggi.

3. **Performa Model Machine Learning (Binary Classification):**
   - Model terbaik: **Logistic Regression** dengan hyperparameter tuning (C=10, solver=lbfgs)
   - Target: Binary — `1 = Dropout`, `0 = Graduate`
   - Accuracy: **93.25%**
   - F1-Score (Weighted): **93.28%**
   - ROC-AUC: **97.37%**
   - Recall Dropout: **94.01%** (model berhasil mendeteksi ~94% dari seluruh siswa dropout)

4. **Perbandingan Model:**

   | Model | Accuracy | F1 (Weighted) | Recall Dropout |
   |-------|----------|----------------|----------------|
   | Logistic Regression (Tuned) ⭐ | **93.25%** | **93.28%** | **94.01%** |
   | Gradient Boosting | 92.84% | 92.85% | 92.25% |
   | Random Forest | 92.01% | 92.02% | 90.14% |

### Rekomendasi Action Items

Berdasarkan hasil analisis data dan model machine learning, berikut rekomendasi yang dapat diimplementasikan oleh Jaya Jaya Institut:

1. **🎯 Implementasi Early Warning System**
   - Deploy model prediksi dropout sebagai sistem deteksi dini yang berjalan setiap akhir semester.
   - Siswa dengan probabilitas dropout > 60% segera ditandai untuk intervensi.
   - Data siswa Enrolled dapat diprediksi menggunakan model yang sama untuk antisipasi dini.

2. **📚 Program Bimbingan Akademik Intensif**
   - Fokus pada siswa dengan approval rate rendah (< 50%) di semester 1.
   - Sediakan kelas remedial dan tutor tambahan untuk mata kuliah dengan tingkat kegagalan tinggi.
   - Target: Meningkatkan rata-rata mata kuliah approved dari 2.5 menjadi minimal 4 per semester untuk siswa berisiko.

3. **💰 Perluasan Program Bantuan Finansial**
   - Perluas cakupan beasiswa, terutama untuk siswa yang menunjukkan kinerja akademik baik namun memiliki kendala finansial.
   - Tawarkan opsi cicilan fleksibel untuk siswa dengan tunggakan SPP.
   - Prioritaskan bantuan untuk siswa berstatus debtor yang masih menunjukkan usaha akademik.

4. **🏫 Perhatian Khusus pada Program Studi Berisiko Tinggi**
   - Program Biofuel Tech (66.7% dropout), Equinculture (55.3%), dan Informatics Engineering (54.1%) memerlukan evaluasi kurikulum dan metode pengajaran.
   - Pertimbangkan penambahan program orientasi dan mentoring peer-to-peer di program-program tersebut.

5. **👥 Program Mentoring dan Konseling**
   - Sediakan program mentoring oleh mahasiswa senior untuk mahasiswa baru, terutama yang berusia di atas 25 tahun (kelompok usia dengan risiko dropout lebih tinggi).
   - Lakukan konseling rutin setiap bulan untuk siswa yang teridentifikasi berisiko.

6. **📊 Monitoring Berkala dengan Dashboard**
   - Gunakan dashboard yang telah dibuat untuk monitoring performa siswa secara real-time.
   - Lakukan review data setiap akhir semester untuk memperbarui strategi intervensi.

