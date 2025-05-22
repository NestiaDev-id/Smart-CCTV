# src/data/split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data_into_sets(df, test_size=0.2, val_size=0.1, random_state=42, stratify_col=None):
    """
    Membagi DataFrame menjadi set pelatihan, validasi, dan pengujian.
    val_size adalah proporsi dari keseluruhan data.
    Mengembalikan train_df, val_df, test_df.
    """
    print("Memulai pembagian data...")

    if df.empty:
        print("DataFrame input kosong, tidak bisa melakukan pembagian.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Cek apakah kolom stratifikasi ada dan valid
    stratify_array = None
    if stratify_col:
        if stratify_col not in df.columns:
            print(f"Peringatan: Kolom stratifikasi '{stratify_col}' tidak ditemukan. Pembagian acak akan dilakukan.")
        elif df[stratify_col].nunique() <= 1 or df[stratify_col].value_counts().min() < 2:
            # Kondisi untuk stratifikasi yang valid: lebih dari 1 kelas, dan setiap kelas punya minimal 2 sampel
            # (atau sesuai kebutuhan train_test_split, biasanya minimal 2 untuk setiap kelas di setiap split)
            print(f"Peringatan: Tidak cukup sampel atau hanya satu kelas unik di kolom '{stratify_col}' untuk stratifikasi yang efektif. Pembagian acak akan dilakukan.")
        else:
            stratify_array = df[stratify_col]
            
    # Validasi ukuran split
    if not (0.0 < test_size < 1.0):
        print("Peringatan: test_size harus antara 0.0 dan 1.0. Menggunakan default 0.2.")
        test_size = 0.2
    if not (0.0 <= val_size < 1.0): # val_size bisa 0
        print("Peringatan: val_size harus antara 0.0 dan 1.0. Menggunakan default 0.1.")
        val_size = 0.1
    if test_size + val_size >= 1.0:
        print("Error: Jumlah test_size dan val_size harus kurang dari 1.0.")
        # Atur ulang ke nilai default yang aman atau kembalikan error
        test_size = 0.2
        val_size = 0.1
        print(f"Menggunakan default: test_size={test_size}, val_size={val_size}")


    # Pembagian pertama: train_val dan test
    # Jika stratify_array valid, gunakan. Jika tidak, train_test_split akan menangani stratify=None.
    # Periksa apakah data cukup untuk test_split
    if len(df) < 2 : # Minimal 2 sampel untuk bisa split
        print("Peringatan: Tidak cukup data untuk melakukan pembagian test/train. Mengembalikan data asli sebagai train_df.")
        return df.copy(), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    try:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_array
        )
    except ValueError as e: # Menangkap error jika stratifikasi gagal (misal, karena kelas yang terlalu kecil)
        print(f"Error saat membagi train/test dengan stratifikasi: {e}. Mencoba tanpa stratifikasi.")
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=None # Fallback tanpa stratifikasi
        )

    # Pembagian kedua: train dan val dari train_val_df
    train_df = pd.DataFrame(columns=df.columns) # Inisialisasi
    val_df = pd.DataFrame(columns=df.columns)   # Inisialisasi

    if val_size > 0 and not train_val_df.empty:
        # Hitung ukuran validasi relatif terhadap sisa setelah test split
        # Ini penting agar proporsi val_size adalah dari dataset *original*
        # (Ukuran train_val) * val_size_relative = (Ukuran original * val_size)
        # val_size_relative = (Ukuran original * val_size) / (Ukuran train_val)
        # Ukuran train_val = Ukuran original * (1 - test_size)
        # Jadi, val_size_relative = (Ukuran original * val_size) / (Ukuran original * (1-test_size)) = val_size / (1-test_size)
        
        if (1 - test_size) == 0: # Mencegah pembagian dengan nol
            print("Peringatan: test_size adalah 1, tidak ada data tersisa untuk validation set.")
            train_df = train_val_df # Semua sisa data menjadi train
        elif len(train_val_df) < 2: # Jika sisa data kurang dari 2, tidak bisa split lagi
            print("Peringatan: Tidak cukup data di train_val_df untuk split validasi. Semua sisa data menjadi train_df.")
            train_df = train_val_df
        else:
            val_size_relative = val_size / (1 - test_size)
            if val_size_relative >= 1.0 : # Jika val_size_relative >= 1, berarti semua sisa data akan jadi val, atau error
                 print(f"Peringatan: val_size ({val_size}) terlalu besar relatif terhadap sisa data setelah test_split. Menyesuaikan val_size_relative agar train_df tidak kosong.")
                 val_size_relative = 0.25 # Ambil 25% dari sisa sebagai val, atau angka lain yang masuk akal
                 if val_size_relative >=1.0 : val_size_relative = 0.0 # Jika masih salah, jangan ada val set.

            stratify_array_tv = None
            if stratify_array is not None and stratify_col in train_val_df.columns:
                if train_val_df[stratify_col].nunique() > 1 and train_val_df[stratify_col].value_counts().min() >= 2:
                     stratify_array_tv = train_val_df[stratify_col]
                else:
                    print(f"Peringatan: Tidak cukup sampel di train_val untuk stratifikasi pada kolom '{stratify_col}' untuk split train/val.")
            
            if val_size_relative > 0 and val_size_relative < 1.0:
                try:
                    train_df, val_df = train_test_split(
                        train_val_df,
                        test_size=val_size_relative,
                        random_state=random_state,
                        stratify=stratify_array_tv
                    )
                except ValueError as e:
                    print(f"Error saat membagi train/val dengan stratifikasi: {e}. Mencoba tanpa stratifikasi untuk train/val.")
                    train_df, val_df = train_test_split(
                        train_val_df,
                        test_size=val_size_relative,
                        random_state=random_state,
                        stratify=None # Fallback
                    )
            else: # Jika val_size_relative adalah 0 atau >= 1 (sudah ditangani)
                train_df = train_val_df # Semua sisa menjadi train jika val_size_relative 0
    elif not train_val_df.empty: # Jika val_size = 0
        train_df = train_val_df
        # val_df sudah diinisialisasi sebagai DataFrame kosong
    
    print(f"  Ukuran data latih: {len(train_df)} baris")
    print(f"  Ukuran data validasi: {len(val_df)} baris")
    print(f"  Ukuran data uji: {len(test_df)} baris")
    
    return train_df, val_df, test_df

# Tidak ada lagi if __name__ == "__main__":
# Fungsi di atas bisa langsung diimpor dan digunakan di notebook.