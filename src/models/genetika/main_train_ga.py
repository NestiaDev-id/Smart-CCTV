# main_train_ga.py
import numpy as np
from genetika import algoritma_genetika_yolo_cnn_lstm
import os

# --- Konfigurasi dan Persiapan Data ---
def load_and_prepare_data(seq_length, frame_height, frame_width):

    print("Memuat dan mempersiapkan data (Placeholder)...")
    # Contoh data dummy 
    num_samples_train = 100
    num_samples_val = 20
    num_classes = 3
    
    train_X = np.random.rand(num_samples_train, seq_length, frame_height, frame_width, 3)
    train_y = np.random.randint(0, num_classes, num_samples_train)
    train_y = np.eye(num_classes)[train_y] # One-hot encoding

    val_X = np.random.rand(num_samples_val, seq_length, frame_height, frame_width, 3)
    val_y = np.random.randint(0, num_classes, val_X.shape[0])
    val_y = np.eye(num_classes)[val_y]
    
    input_shape_per_frame = (frame_height, frame_width, 3)
    
    return train_X, train_y, val_X, val_y, input_shape_per_frame, num_classes

if __name__ == "__main__":
    # --- Parameter Algoritma Genetika ---
    JUMLAH_KROMOSOM = 10  # Ukuran populasi
    GENERATIONS = 5      # Jumlah generasi
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    EPOCHS_PER_EVAL = 5 # Berapa epoch untuk melatih setiap individu
    
    # Ukuran frame yang Anda gunakan untuk training
    FRAME_HEIGHT = 64
    FRAME_WIDTH = 64
    
    # Asumsi seq_length terpanjang dari hyperparameter Anda
    # Ini penting agar data yang dimuat cocok
    MAX_SEQ_LENGTH = 30 
    
    print("--- Memulai Proses Optimasi Hyperparameter dengan Algoritma Genetika ---")
    
    # 1. Muat Data
    train_X, train_y, val_X, val_y, input_shape, num_classes = load_and_prepare_data(
        MAX_SEQ_LENGTH, FRAME_HEIGHT, FRAME_WIDTH
    )
    
    # 2. Jalankan Algoritma Genetika
    best_hyperparams_found, fitness_history = algoritma_genetika_yolo_cnn_lstm(
        train_X=train_X, train_y=train_y,
        val_X=val_X, val_y=val_y,
        input_shape_per_frame=input_shape,
        num_classes=num_classes,
        jumlah_kromosom=JUMLAH_KROMOSOM,
        generations=GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        epochs_per_eval=EPOCHS_PER_EVAL
    )
    
    if best_hyperparams_found:
        print("\n--- Optimasi Selesai ---")
        print("Hyperparameter terbaik yang ditemukan:")
        print(best_hyperparams_found['hyperparameters'])
        print(f"Dengan Fitness (Val Accuracy) terbaik: {best_hyperparams_found['fitness']:.4f}")
        # Di sini Anda bisa melatih model final dengan hyperparameter terbaik dan epoch lebih banyak,
        # lalu menyimpan bobotnya.
    else:
        print("\nOptimasi selesai, namun tidak ada individu yang baik ditemukan.")