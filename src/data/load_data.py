# src/data/load_data.py
import cv2
import os
import numpy as np

def load_images_from_folder(folder_path):
    """
    Memuat semua gambar dari folder yang diberikan.
    
    Parameters:
    - folder_path (str): Path ke folder yang berisi gambar-gambar mobil.
    
    Returns:
    - images (list): Daftar gambar dalam folder.
    """
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Error: Gambar {filename} tidak dapat dimuat.")
    return images


def load_video(video_path):
    """
    Memuat video dari path yang diberikan dan mengembalikan frame-frame dalam video.
    
    Parameters:
    - video_path (str): Lokasi file video yang akan dimuat.
    
    Returns:
    - frames (list): Daftar frame dalam video.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video dari {video_path}")
        return []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

# Contoh penggunaan
if __name__ == "__main__":
    # Memuat gambar-gambar mobil dari folder
    folder_path = 'data/raw/mobil'
    images = load_images_from_folder(folder_path)
    print(f"Jumlah gambar mobil yang dimuat: {len(images)}")

    # Memuat video mobil
    video_path = 'data/raw/rekaman-mobil.mp4'
    frames = load_video(video_path)
    print(f"Jumlah frame dalam video: {len(frames)}")
