# 🐟 Analisis Performa Melalui Optimasi ADAM dan GRAD-CAM dengan ResNet50 pada Klasifikasi Multi-Kelas Ikan

> Menyelesaikan Masalah Black-Box pada Klasifikasi Ikan melalui Explainable AI (XAI)

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://klasifikasiikan.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)
![Keras](https://img.shields.io/badge/Keras-3.14.1-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Deskripsi

Proyek ini tidak hanya berfokus pada pencapaian akurasi tinggi dalam klasifikasi 9 spesies ikan menggunakan arsitektur **ResNet50** dan **Adam Optimizer**, tetapi secara khusus memecahkan masalah krisis transparansi (*black-box*) pada model *Deep Learning*. 

Dengan mengintegrasikan **Explainable Artificial Intelligence (XAI)** melalui algoritma **Grad-CAM** (Gradient-weighted Class Activation Mapping), sistem ini mampu memvisualisasikan *heatmap* yang melokalisasi area fokus model. Hal ini membuktikan secara empiris bahwa keputusan model didasarkan pada fitur morfologi biologis ikan yang relevan (seperti struktur sirip, pola warna, dan bentuk kepala), bukan akibat mengeksploitasi bias pada artefak latar belakang citra.

Aplikasi ini dibangun sebagai luaran tugas akhir/skripsi dan dapat diakses secara langsung melalui antarmuka web interaktif berbasis **Streamlit**.

---

## 🚀 Demo

🌐 **Live App:** [klasifikasiikan.streamlit.app](https://klasifikasiikan.streamlit.app)

---

## ✨ Fitur

- ✅ Klasifikasi multi-kelas untuk 9 jenis ikan dari gambar input
- **✅ Explainable AI (XAI): Visualisasi Grad-CAM untuk transparansi dan interpretabilitas keputusan model**
- ✅ Transfer Learning menggunakan ResNet50 (pretrained ImageNet)
- ✅ Optimasi model dengan **Adam Optimizer** dan *Adaptive Learning Rate* (ReduceLROnPlateau)
- ✅ Antarmuka web interaktif berbasis Streamlit
- ✅ Menampilkan **Top 3** probabilitas prediksi per kelas
- ✅ Riwayat prediksi (History)

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Framework ML | TensorFlow 2.21 / Keras 3.14 |
| Arsitektur Model | ResNet50 (Transfer Learning) |
| Optimizer | Adam |
| Interpretabilitas | Grad-CAM (XAI) |
| Web App | Streamlit |
| Image Processing | Pillow, NumPy |
| Bahasa | Python 3.12 |

---

## 🐠 Kelas Ikan

Model ini mampu mengklasifikasikan **9 jenis ikan** laut berikut:

| No | Kelas | No | Kelas |
|----|-------|----|-------|
| 1 | Black Sea Sprat | 6 | Sea Bass |
| 2 | Gilt-Head Bream | 7 | Shrimp |
| 3 | Horse Mackerel | 8 | Striped Red Mullet |
| 4 | Red Mullet | 9 | Trout |
| 5 | Red Sea Bream | | |

---

## 📊 Performa Model (Adam + Pre-trained ResNet50)

Model utama dilatih menggunakan **Adam optimizer** dengan **ResNet50 pre-trained ImageNet** sebagai *backbone* (frozen). Training konvergen dengan sangat cepat dan dihentikan pada **epoch 10** menggunakan mekanisme *Early Stopping*.

### Hasil Evaluasi (Data Uji / Testing)

| Metrik | Nilai |
|--------|-------|
| **Akurasi Testing** | **99.94%** |
| **Loss Testing** | **0.00412** |
| Precision (Macro Avg) | 1.00 |
| Recall (Macro Avg) | 1.00 |
| F1-Score (Macro Avg) | 1.00 |
| Waktu Pelatihan | 15,15 menit (10 epoch) |

---

## 🔍 Visualisasi Interpretabilitas (Grad-CAM)

Untuk memvalidasi bahwa model tidak sekadar menebak berdasarkan latar belakang citra, metode **Grad-CAM** diterapkan pada lapisan konvolusi terakhir (`conv5_block3_out`). 

<img width="1340" height="990" alt="gradcam" src="https://github.com/user-attachments/assets/d3593c17-8727-46d1-8045-26c0e11cf16e" />

*Area berwarna merah menandai fitur taksonomi dengan bobot aktivasi paling tinggi.* Visualisasi membuktikan bahwa model secara konsisten memusatkan atensinya pada struktur anatomi diskriminatif (seperti bentuk badan, operkulum, dan sirip) sebagai dasar keputusan klasifikasi.

*(**Catatan untuk Daffa:** Jangan lupa ganti teks ini dengan memasukkan 1 atau 2 screenshot contoh gambar Grad-CAM dari skripsimu, misalnya: `<img src="link-gambar-gradcam.png" alt="Contoh Grad-CAM">`)*

---

## 🔬 Eksperimen Perbandingan (Ablation Study)

Untuk memvalidasi signifikansi dari transfer learning dan optimizer Adam, sebuah eksperimen pembanding (Model B) dilakukan dengan melatih seluruh jaringan ResNet50 dari nol menggunakan inisialisasi acak dan optimizer SGD.

### Ringkasan Perbandingan Performa

| Aspek / Metrik | Model A (Utama) | Model B (Pembanding) |
|----------------|-----------------|----------------------|
| **Weights** | Pre-trained ImageNet | Random Init (dari nol) |
| **Optimizer** | Adam | SGD |
| **Akurasi Testing** | **99.94%** | 99.22% |
| **Loss Testing** | **0.00412** | 0.02292 |
| **Konvergensi** | Konvergen Cepat (10 Epoch) | Lebih Lambat (20 Epoch) |
| **Kesalahan Prediksi** | Hanya 1 sampel salah (0.06%) | Tersebar di banyak kelas (0.78%) |

**Kesimpulan:** Penggunaan transfer learning dengan optimasi Adam terbukti secara empiris mampu menavigasi ruang parameter lebih efisien, memangkas waktu komputasi, dan menghasilkan klasifikasi yang jauh lebih solid dibandingkan model konvensional tanpa pra-latih.

---

## 📁 Struktur Folder

```
deploy/
├── fish.py                  # File utama aplikasi Streamlit
├── requirements.txt         # Daftar dependencies
├── runtime.txt              # Versi Python untuk Streamlit Cloud
├── .gitignore               # File yang dikecualikan dari Git
├── .streamlit/
│   └── config.toml          # Konfigurasi tampilan Streamlit (dark theme)
├── banner.png               # Gambar banner aplikasi
├── logo.png                 # Logo aplikasi
├── README.md                # Dokumentasi proyek
└── *.jpeg / *.png           # Gambar contoh per kelas ikan
```

> ⚠️ File `model.h5` tidak disertakan di repository karena ukurannya melebihi batas GitHub (100 MB). Model akan diunduh otomatis dari Google Drive saat aplikasi pertama kali dijalankan.

---

## 📥 Download Model

Model `.h5` tersedia di Google Drive:

🔗 **[Download model.h5](https://drive.google.com/file/d/1vAijkvT2TrxjSS-GvpNfZLy3he4aGGv9/view?usp=drive_link)**

Setelah diunduh, letakkan file `model.h5` di root folder project.

---

## ⚙️ Cara Menjalankan Secara Lokal

**1. Clone repository**
```bash
git clone https://github.com/daffahanif6/resnet50-fish-classifier-adam-gradcam.git
cd resnet50-fish-classifier-adam-gradcam
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download model** dan letakkan `model.h5` di folder root (atau biarkan aplikasi mengunduh otomatis saat pertama kali dijalankan).

**4. Jalankan aplikasi**
```bash
streamlit run fish.py
```

---

## 🗂️ Dataset

- **Nama**: [A Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) (Kaggle)
- **Jumlah Gambar**: 9.000 gambar (1.000 per kelas)
- **Jumlah Kelas**: 9 jenis ikan
- **Pembagian**: 80% training (dengan 20% validasi), 20% testing
- **Preprocessing**: ResNet50 `preprocess_input`, resize ke 224×224

---

## 👤 Author

| | |
|---|---|
| **Nama** | Daffa Hanif Durachman |
| **NIM** | 21.11.4470 |
| **Universitas** | Universitas Amikom Yogyakarta |
| **Program Studi** | Informatika |
| **Tahun** | 2026 |

---

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE).
