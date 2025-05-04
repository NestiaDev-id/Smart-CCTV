# src/data/load_data.py
import cv2
import os

def load_video(video_path):
    """
    Memuat video dari path yang diberikan dan mengembalikan frame frame dalam video.
    
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

def load_image(image_path):
    """
    Memuat gambar dari path yang diberikan.
    
    Parameters:
    - image_path (str): Lokasi file gambar.
    
    Returns:
    - frame (ndarray): Gambar yang dimuat dalam bentuk array.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Gambar tidak dapat dimuat dari {image_path}")
    return frame

# Contoh penggunaan
if __name__ == "__main__":
    video_path = 'data/raw/video.mp4'
    frames = load_video(video_path)
    print(f"Jumlah frame dalam video: {len(frames)}")
