import pandas as pd
import os

def clean_missing_values(df, columns_to_check=None):
    """
    Menangani nilai yang hilang.
    """
    print("Memeriksa nilai yang hilang...")
    initial_rows = len(df)
    
    # Buat salinan untuk menghindari SettingWithCopyWarning saat memodifikasi DataFrame
    df_cleaned = df.copy()

    if columns_to_check:
        df_cleaned.dropna(subset=columns_to_check, inplace=True)
    else:
        df_cleaned.dropna(inplace=True)
        
    rows_after_na = len(df_cleaned)
    print(f"  {initial_rows - rows_after_na} baris dihapus karena nilai yang hilang.")
    return df_cleaned

def remove_duplicates(df, subset_cols=None):
    """
    Menghapus baris duplikat.
    """
    print("Menghapus duplikat...")
    initial_rows = len(df)
    df_cleaned = df.copy()
    df_cleaned.drop_duplicates(subset=subset_cols, keep='first', inplace=True)
    rows_after_duplicates = len(df_cleaned)
    print(f"  {initial_rows - rows_after_duplicates} baris duplikat dihapus.")
    return df_cleaned

def filter_data_by_condition(df, column_name, condition_value, operator="=="):
    """
    Memfilter data berdasarkan kondisi tertentu.
    """
    print(f"Memfilter data: Kolom '{column_name}' {operator} '{condition_value}'...")
    initial_rows = len(df)
    df_filtered = df.copy() # Bekerja pada salinan

    if column_name not in df_filtered.columns:
        print(f"  Peringatan: Kolom '{column_name}' tidak ditemukan dalam DataFrame. Tidak ada filter yang diterapkan.")
        return df_filtered

    try:
        if operator == "==":
            df_filtered = df_filtered[df_filtered[column_name] == condition_value]
        elif operator == "!=":
            df_filtered = df_filtered[df_filtered[column_name] != condition_value]
        elif operator == ">":
            # Pastikan tipe data mendukung perbandingan numerik
            if pd.api.types.is_numeric_dtype(df_filtered[column_name]):
                df_filtered = df_filtered[df_filtered[column_name] > condition_value]
            else:
                print(f"  Peringatan: Kolom '{column_name}' bukan numerik. Filter '>' tidak bisa diterapkan.")
        elif operator == "<":
            if pd.api.types.is_numeric_dtype(df_filtered[column_name]):
                df_filtered = df_filtered[df_filtered[column_name] < condition_value]
            else:
                print(f"  Peringatan: Kolom '{column_name}' bukan numerik. Filter '<' tidak bisa diterapkan.")
        # Tambahkan operator lain jika perlu
        else:
            print(f"  Operator '{operator}' tidak didukung. Tidak ada filter yang diterapkan.")
            return df_filtered
    except Exception as e:
        print(f"  Error saat menerapkan filter pada kolom '{column_name}': {e}")
        return df # Kembalikan DataFrame asli jika ada error saat filter

    rows_after_filter = len(df_filtered)
    print(f"  {initial_rows - rows_after_filter} baris dihapus oleh filter.")
    return df_filtered

def check_file_paths_exist(df, path_column='video_path'):
    """
    Memeriksa apakah path file dalam kolom tertentu ada.
    Menghapus baris jika path tidak ada.
    """
    if path_column not in df.columns:
        print(f"Peringatan: Kolom path '{path_column}' tidak ditemukan untuk pemeriksaan keberadaan file.")
        return df

    print(f"Memeriksa keberadaan file di kolom '{path_column}'...")
    initial_rows = len(df)
    # Pastikan path adalah string sebelum os.path.exists
    df_checked = df[df[path_column].apply(lambda x: isinstance(x, str) and os.path.exists(x))]
    rows_after_check = len(df_checked)
    if initial_rows - rows_after_check > 0:
        print(f"  {initial_rows - rows_after_check} baris dihapus karena path file tidak valid/tidak ditemukan.")
    else:
        print(f"  Semua path file di kolom '{path_column}' valid.")
    return df_checked