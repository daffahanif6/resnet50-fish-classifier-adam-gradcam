# 🐟 Klasifikasi Multi-Kelas Ikan Menggunakan Transfer Learning dengan ResNet50 dan Optimasi Adam

> Analisis Performa dan Interpretabilitas Model melalui Grad-CAM

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://klasifikasiikan.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Deskripsi

Proyek ini merupakan implementasi sistem klasifikasi ikan berbasis deep learning menggunakan arsitektur **ResNet50** dengan pendekatan **Transfer Learning**. Model dilatih menggunakan optimizer **Adam** dan dilengkapi dengan visualisasi interpretabilitas menggunakan **Grad-CAM** (Gradient-weighted Class Activation Mapping) untuk menunjukkan area gambar yang menjadi fokus model dalam melakukan prediksi.

Aplikasi ini dibangun sebagai tugas akhir/skripsi dan dapat diakses secara langsung melalui antarmuka web berbasis **Streamlit**.

---

## 🚀 Demo

🌐 **Live App:** [klasifikasiikan.streamlit.app](https://klasifikasiikan.streamlit.app)

---

## ✨ Fitur

- ✅ Klasifikasi multi-kelas ikan dari gambar input
- ✅ Transfer Learning menggunakan ResNet50 (pretrained ImageNet)
- ✅ Optimasi model dengan Adam Optimizer
- ✅ Visualisasi Grad-CAM untuk interpretabilitas model
- ✅ Antarmuka web interaktif berbasis Streamlit
- ✅ Menampilkan probabilitas prediksi per kelas

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Framework ML | TensorFlow / Keras |
| Arsitektur Model | ResNet50 |
| Optimizer | Adam |
| Interpretabilitas | Grad-CAM |
| Web App | Streamlit |
| Image Processing | OpenCV, Pillow |
| Bahasa | Python 3.11 |

---

## 🐠 Kelas Ikan

Model ini mampu mengklasifikasikan beberapa jenis ikan, di antaranya:

- Black Sea Sprat
- Gilt-Head Bream
- Horse Mackerel
- Red Mullet
- Red Sea Bream
- Sea Bass
- Shrimp
- Striped Red Mullet
- Trout

---

## 📁 Struktur Folder

```
deploying/
├── fish.py                  # File utama aplikasi Streamlit
├── requirements.txt         # Daftar dependencies
├── runtime.txt              # Versi Python untuk Streamlit Cloud
├── .gitignore               # File yang dikecualikan dari Git
├── .streamlit/
│   └── config.toml          # Konfigurasi tampilan Streamlit
├── banner.png               # Gambar banner aplikasi
├── logo.png                 # Logo aplikasi
└── *.jpeg / *.png           # Gambar contoh per kelas ikan
```

> ⚠️ File `model.h5` tidak disertakan di repository karena ukurannya melebihi batas GitHub (100 MB). Model dapat diunduh melalui link di bawah.

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

**3. Download model** dan letakkan `model.h5` di folder root.

**4. Jalankan aplikasi**
```bash
streamlit run fish.py
```

---

## 📊 Performa Model

| Metrik | Nilai |
|--------|-------|
| Akurasi Training (Epoch Terakhir) | 99.95% |
| Akurasi Validasi (Epoch Terakhir) | 99.93% |
| Akurasi Testing | **100.00%** |
| Loss Training (Epoch Terakhir) | 0.0022 |
| Loss Validasi (Epoch Terakhir) | 0.0014 |
| Loss Testing | 0.00102 |

---

## 👤 Author

| | |
|---|---|
| **Nama** | Daffa Hanif |
| **Universitas** | Universitas Amikom Yogyakarta |
| **Program Studi** | Informatika |
| **Tahun** | 2026 |

---

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE).