from ultralytics import YOLO
import os

model_path = "yolov8n.pt" # Atau path absolut jika tidak di direktori kerja

# Cek apakah file ada sebelum mencoba memuat
if not os.path.exists(model_path):
    print(f"Error: File model '{model_path}' tidak ditemukan.")
    # Anda bisa mencoba memuat dengan YOLO("yolov8n.pt") agar otomatis download jika belum ada
    # atau jika Anda ingin memastikan file yang sudah ada itu valid.
else:
    print(f"File model '{model_path}' ditemukan. Mencoba memuat...")
    try:
        # Muat model
        model = YOLO(model_path)
        print(f"Model '{model_path}' berhasil dimuat.")
        
        # Tampilkan beberapa informasi tentang model (opsional)
        print("\nInformasi Model:")
        # model.info() # Metode info() mungkin memberikan ringkasan yang baik
        
        # Anda juga bisa melihat nama kelas yang dikenali oleh model (untuk COCO dataset)
        if hasattr(model, 'names'):
            print(f"  Jumlah kelas: {len(model.names)}")
            print(f"  Contoh nama kelas: {list(model.names.values())[:5]}") # Tampilkan 5 kelas pertama
        else:
            # Untuk beberapa versi/tugas, model.names mungkin tidak langsung ada
            # Coba yolo_results.names setelah melakukan prediksi
            pass

        # (Opsional) Lakukan prediksi pada gambar contoh untuk memastikan fungsionalitas dasar
        # Anda memerlukan gambar contoh untuk ini
        # sample_image = "path/to/your/test_image.jpg" 
        # if os.path.exists(sample_image):
        #     print(f"\nMencoba prediksi pada gambar: {sample_image}")
        #     results = model(sample_image, verbose=False)
        #     print(f"Prediksi berhasil. Ditemukan {len(results[0].boxes)} objek.")
        # else:
        #     print(f"\nFile gambar contoh '{sample_image}' tidak ditemukan untuk pengujian prediksi.")

        print("\nPengecekan dasar model YOLO berhasil.")

    except Exception as e:
        print(f"Error saat memuat atau menggunakan model '{model_path}': {e}")