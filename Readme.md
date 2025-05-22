# Optimasi Model YOLO-CNN-LSTM Menggunakan Algoritma Genetika untuk Analisis Perilaku Kendaraan pada Sistem CCTV Cerdas

Proyek ini bertujuan untuk membangun sistem analisis video cerdas menggunakan deteksi objek YOLOv11, ekstraksi fitur dengan CNN, analisis sekuensial dengan LSTM, dan optimasi hyperparameter menggunakan Algoritma Genetika (GA).

## Fitur Utama

- **Deteksi Objek**: Menggunakan YOLOv11 untuk mendeteksi objek (misalnya, kendaraan) dalam frame video.
- **Ekstraksi Fitur**: CNN digunakan untuk mengekstrak fitur visual dari objek yang terdeteksi.
- **Analisis Sekuensial**: LSTM memproses fitur sekuensial dari objek yang dilacak untuk analisis temporal (misalnya, klasifikasi perilaku).
- **Optimasi Hyperparameter**: Algoritma Genetika (GA) digunakan untuk menemukan hyperparameter optimal untuk model CNN-LSTM.

## Struktur Proyek

Berikut adalah gambaran umum struktur direktori proyek ini:

```bash
├── data/                    # Direktori untuk dataset mentah, diproses, dll.
├── notebooks/               # Jupyter notebooks untuk eksperimen dan analisis
│   ├── 00-Experiment.ipynb
│   ├── 01-data-exploration.ipynb
│   ├── 02-prepocessing.ipynb
│   └── ...
├── src/                     # Kode sumber utama
│   ├── data/                # Skrip untuk pemrosesan data
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   └── split_data.py
│   ├── models/              # Skrip untuk definisi model, pelatihan, dan evaluasi
│   │   ├── cnn/
│   │   │   └── model.py     # Definisi arsitektur CNN
│   │   ├── yolo/
│   │   │   └── track.py     # Skrip untuk tracking objek dengan YOLO
│   │   ├── genetika/        # Implementasi Algoritma Genetika
│   │   │   ├── buildModel.py
│   │   │   ├── crossover.py
│   │   │   ├── evaluasi_fitness.py
│   │   │   ├── genetika.py
│   │   │   ├── main.py      # Skrip utama untuk menjalankan GA
│   │   │   ├── mutation.py
│   │   │   ├── pembentukan_populasi.py
│   │   │   └── selection.py
│   │   └── lstm/            # Definisi arsitektur LSTM atau CNN-LSTM gabungan
│   ├── utils/               # Fungsi utilitas (Opsional)
│   └── main.py              # Skrip utama untuk menjalankan alur kerja
├── backend/                 # Merupakan backend untuk fast api
├── frontend/                # Merupakan frontend untuk react+vite+tyscript
├── requirements.txt         # Daftar dependensi Python
└── README.md

```

## Prasyarat

- Python 3.8+
- Pip (Python package installer)

## Instalasi

Untuk menyiapkan lingkungan dan menjalankan proyek ini, ikuti langkah-langkah berikut:

1.  **Klon Repositori (jika sudah diunggah ke Git)**:

    ```bash
    git clone https://github.com/NestiaDev-id/Smart-CCTV.git
    cd Smart-CCTV
    ```

2.  **Buat dan Aktifkan Virtual Environment (Direkomendasikan)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/Mac
    # venv\Scripts\activate    # Untuk Windows
    ```

3.  **Instal Dependensi**:
    Pastikan Anda memiliki Python [versi, misal: 3.8+] terinstal. Kemudian jalankan:

    ```bash
    pip install -r requirements.txt
    ```

    File `requirements.txt` berisi semua pustaka Python yang dibutuhkan, termasuk `torch`, `tensorflow`, `ultralytics`, `opencv-python`, dll.

## Penggunaan

### 1. Persiapan Data

- Tempatkan dataset Anda di direktori `data/`.
- Jalankan skrip pra-pemrosesan yang relevan dari `src/data/` atau ikuti langkah-langkah di notebook `notebooks/01-data-exploration.ipynb` dan `notebooks/02-prepocessing.ipynb`.
  ```bash
  # Contoh
  # python src/data/preprocess_script.py --input data/raw --output data/processed
  ```
