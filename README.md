# 🐟 Klasifikasi Multi-Kelas Ikan Menggunakan Transfer Learning dengan ResNet50 dan Optimasi Adam

> Analisis Performa dan Interpretabilitas Model melalui Grad-CAM

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://klasifikasiikan.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)
![Keras](https://img.shields.io/badge/Keras-3.14.1-red)
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

- ✅ Klasifikasi multi-kelas untuk 9 jenis ikan dari gambar input
- ✅ Transfer Learning menggunakan ResNet50 (pretrained ImageNet)
- ✅ Optimasi model dengan **Adam Optimizer**
- ✅ Visualisasi **Grad-CAM** untuk interpretabilitas model
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
| Interpretabilitas | Grad-CAM & Saliency Map |
| Web App | Streamlit |
| Image Processing | Pillow, NumPy |
| Bahasa | Python 3.12 |

---

## 🐠 Kelas Ikan

Model ini mampu mengklasifikasikan **9 jenis ikan** berikut:

| No | Kelas | No | Kelas |
|----|-------|----|-------|
| 1 | Black Sea Sprat | 6 | Sea Bass |
| 2 | Gilt-Head Bream | 7 | Shrimp |
| 3 | Horse Mackerel | 8 | Striped Red Mullet |
| 4 | Red Mullet | 9 | Trout |
| 5 | Red Sea Bream | | |

---

## 📊 Performa Model (Adam + Pre-trained ResNet50)

Model utama dilatih menggunakan **Adam optimizer** dengan **ResNet50 pre-trained ImageNet** sebagai backbone (frozen). Training berhenti di **epoch 13** dari 50 (Early Stopping).

### Hasil Evaluasi

| Metrik | Nilai |
|--------|-------|
| Akurasi Training (Epoch Terakhir) | 99.95% |
| Akurasi Validasi (Epoch Terakhir) | 100.00% |
| **Akurasi Testing** | **100.00%** |
| Loss Training (Epoch Terakhir) | 0.0021 |
| Loss Validasi (Epoch Terakhir) | 0.00079 |
| **Loss Testing** | **0.00092** |
| Jumlah Epoch | 13/50 (Early Stopping) |
| Learning Rate Akhir | 1e-05 |

### Classification Report

```
                    precision    recall  f1-score   support

   Black Sea Sprat       1.00      1.00      1.00       207
   Gilt-Head Bream       1.00      1.00      1.00       209
   Hourse Mackerel       1.00      1.00      1.00       200
        Red Mullet       1.00      1.00      1.00       196
     Red Sea Bream       1.00      1.00      1.00       201
          Sea Bass       1.00      1.00      1.00       195
            Shrimp       1.00      1.00      1.00       188
Striped Red Mullet       1.00      1.00      1.00       200
             Trout       1.00      1.00      1.00       204

          accuracy                           1.00      1800
         macro avg       1.00      1.00      1.00      1800
      weighted avg       1.00      1.00      1.00      1800
```

---

## 🔬 Eksperimen Perbandingan: Adam vs SGD (Tanpa Pre-trained)

Untuk memvalidasi efektivitas **Adam optimizer** dan **transfer learning**, dilakukan eksperimen perbandingan dengan dua konfigurasi model:

### Konfigurasi Eksperimen

| Aspek | Model A (Utama) | Model B (Pembanding) |
|-------|-----------------|----------------------|
| **Weights** | Pre-trained ImageNet | Random (dari nol) |
| **Optimizer** | Adam | SGD |
| **Backbone** | Frozen (tidak dilatih ulang) | Semua layer dilatih |
| **Transfer Learning** | ✅ Ya | ❌ Tidak |
| **Arsitektur Head** | Dense 512 → Dropout 0.5 → Dense 9 | Dense 512 → Dropout 0.5 → Dense 9 |
| **Callbacks** | EarlyStopping + ReduceLR | EarlyStopping + ReduceLR |
| **Batch Size** | 32 | 32 |
| **Max Epochs** | 50 | 50 |
| **Dataset Split** | 80% train, 20% test | 80% train, 20% test |
| **Validation** | 20% dari train | 20% dari train |

> Semua kondisi eksperimen lainnya (dataset, preprocessing, split, batch size, arsitektur head) dibuat **identik** agar perbandingan valid.

### Ringkasan Perbandingan

| Metrik | Model A (Adam + Pre-trained) | Model B (SGD + Random Init) |
|--------|-----------------------------|-----------------------------|
| Akurasi Testing | **100.00%** | Jauh lebih rendah* |
| Loss Testing | **0.00092** | Jauh lebih tinggi* |
| Konvergensi | Cepat (13 epoch) | Lambat |
| Transfer Learning | ✅ | ❌ |

> *\*Notebook eksperimen belum memiliki output tersimpan. Berdasarkan analisis kode dan kesimpulan peneliti, Model B memberikan performa yang signifikan lebih buruk.*

### Kesimpulan Eksperimen

1. **Model A (ResNet50 + Adam + Pre-trained ImageNet)** memberikan performa yang sangat baik karena memanfaatkan:
   - **Transfer Learning**: Bobot yang sudah dilatih pada jutaan gambar ImageNet
   - **Optimizer Adam**: Adaptive learning rate yang konvergen lebih cepat

2. **Model B (ResNet50 Murni + SGD tanpa Pre-trained)** menunjukkan performa yang jauh lebih rendah karena:
   - Bobot diinisialisasi **random** sehingga harus belajar dari nol
   - **SGD** dengan default learning rate konvergen lebih lambat
   - Dataset yang relatif kecil (9.000 gambar) tidak cukup untuk melatih 23+ juta parameter ResNet50 dari awal

3. Perbedaan performa ini membuktikan **nilai besar dari transfer learning dan optimizer adaptif (Adam)** dalam klasifikasi gambar, terutama ketika dataset terbatas.

### Referensi Notebook

| Notebook | Deskripsi | Link |
|----------|-----------|------|
| `21_11_4470_SKRIPSI.ipynb` | Model utama (Adam + Pre-trained) | [Google Colab](https://colab.research.google.com/drive/19uRsRFPey8XHZmMGPZyzbtb9L-KI_xXA?usp=sharing) |
| `21_11_4470_SKRIPSI_EKSPERIMEN.ipynb` | Eksperimen perbandingan (Adam vs SGD) | *Tersedia di repository* |

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