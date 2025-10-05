# backend/app/core/model.py

import torch
from ultralytics import YOLO
# --- Perubahan di sini ---
from deep_sort_realtime.deepsort_tracker import DeepSort

print("🚀 Inisialisasi Core Model...")

# Periksa ketersediaan GPU (CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--> Menggunakan device: {device}")

# Muat model YOLOv8
print("--> Memuat model YOLOv8...")
model = YOLO('weights/yolov8n.pt').to(device)
class_names = model.names
print("--> ✅ Model YOLOv8 berhasil dimuat.")

# Inisialisasi tracker DeepSORT
print("--> Menginisialisasi DeepSORT tracker...")
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
print("--> ✅ Tracker berhasil diinisialisasi.")