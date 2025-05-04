# src/data/split_data.py
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(frames, test_size=0.2, random_state=42):
    """
    Membagi dataset menjadi set pelatihan dan pengujian.
    
    Parameters:
    - frames (list): Daftar frame untuk dibagi.
    - test_size (float): Proporsi data untuk pengujian (default 0.2).
    - random_state (int): Nilai untuk memastikan pembagian data yang konsisten.
    
    Returns:
    - train_frames (list): Daftar frame untuk pelatihan.
    - test_frames (list): Daftar frame untuk pengujian.
    """
    train_frames, test_frames = train_test_split(frames, test_size=test_size, random_state=random_state)
    return train_frames, test_frames

# Contoh penggunaan
if __name__ == "__main__":
    from load_data import load_video
    video_path = 'data/raw/video.mp4'
    frames = load_video(video_path)
    train_frames, test_frames = split_data(frames)
    print(f"Jumlah frame untuk pelatihan: {len(train_frames)}")
    print(f"Jumlah frame untuk pengujian: {len(test_frames)}")
