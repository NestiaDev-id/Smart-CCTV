import os
import pandas as pd
import glob # Untuk mencari file dengan pola tertentu

# --- Fungsi untuk memuat berbagai jenis data ---

def load_video_paths(data_dir):
    """
    Memuat path ke file video dari direktori.
    Asumsi: semua file video relevan ada di dalam data_dir atau subdirektorinya.
    """
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"] # Tambahkan ekstensi lain jika perlu
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    
    if not video_paths:
        print(f"Peringatan: Tidak ada file video yang ditemukan di {data_dir}")
    return video_paths

def load_image_paths(data_dir):
    """
    Memuat path ke file gambar (frame) dari direktori.
    """
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    
    if not image_paths:
        print(f"Peringatan: Tidak ada file gambar yang ditemukan di {data_dir}")
    return image_paths

def load_annotations(annotation_path):
    """
    Memuat anotasi. Ini sangat bergantung pada format anotasi Anda (misalnya, CSV, JSON, XML).
    Contoh untuk file CSV:
    """
    if not os.path.exists(annotation_path):
        print(f"Peringatan: File anotasi tidak ditemukan di {annotation_path}")
        return pd.DataFrame() # Kembalikan DataFrame kosong jika file tidak ada

    try:
        if annotation_path.endswith('.csv'):
            annotations_df = pd.read_csv(annotation_path)
        # Tambahkan parsing untuk format lain seperti JSON atau XML jika perlu
        # elif annotation_path.endswith('.json'):
        #     annotations_df = pd.read_json(annotation_path)
        else:
            print(f"Peringatan: Format anotasi tidak didukung untuk {annotation_path}")
            return pd.DataFrame()
        
        print(f"Berhasil memuat anotasi dari {annotation_path} dengan {len(annotations_df)} baris.")
        return annotations_df
    except Exception as e:
        print(f"Error saat memuat anotasi dari {annotation_path}: {e}")
        return pd.DataFrame()

def process_loaded_data(raw_data_path, annotation_file_path=None):
    """
    Fungsi untuk memuat data mentah dan mengembalikan DataFrame yang diproses.
    Ini menggabungkan logika dari fungsi main sebelumnya tetapi mengembalikan DataFrame.
    """
    print(f"Memproses pemuatan data dari: {raw_data_path}")
    if annotation_file_path:
        print(f"Memuat anotasi dari: {annotation_file_path}")

    video_files = load_video_paths(raw_data_path)
    all_annotations_df = pd.DataFrame()
    if annotation_file_path:
        all_annotations_df = load_annotations(annotation_file_path)

    if not video_files and all_annotations_df.empty:
        print("Tidak ada data video atau anotasi yang dimuat.")
        return pd.DataFrame()

    loaded_data_info = []
    for vid_path in video_files:
        video_filename = os.path.basename(vid_path)
        loaded_data_info.append({'video_path': vid_path, 'filename': video_filename})
    
    loaded_data_df = pd.DataFrame(loaded_data_info)

    if not all_annotations_df.empty and 'filename' in all_annotations_df.columns and not loaded_data_df.empty:
        cols_to_use = all_annotations_df.columns.difference(loaded_data_df.columns.drop('filename'))
        # Jika 'filename' tidak unik di salah satu DataFrame, merge bisa menghasilkan lebih banyak baris dari yang diharapkan.
        # Pertimbangkan validasi atau penanganan duplikat sebelum merge jika ini adalah masalah.
        processed_data_df = pd.merge(loaded_data_df, all_annotations_df[list(cols_to_use) + ['filename']], on='filename', how='left')

        print("Data video dan anotasi digabungkan (jika ada kecocokan).")
    elif not loaded_data_df.empty:
        processed_data_df = loaded_data_df
        print("Hanya data video yang dimuat (tidak ada anotasi atau tidak bisa digabung).")
    elif not all_annotations_df.empty:
        processed_data_df = all_annotations_df
        print("Hanya data anotasi yang dimuat.")
    else:
        print("Tidak ada data yang diproses.")
        return pd.DataFrame()
        
    return processed_data_df

# Tidak ada lagi if __name__ == "__main__":
# Fungsi-fungsi di atas bisa langsung diimpor dan digunakan di notebook.