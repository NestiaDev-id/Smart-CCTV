# backend/app/core/model.py

import torch
from ultralytics import YOLO
from ocsort.ocsort import OCSort

print("🚀 Inisialisasi Core Model...")

# Periksa ketersediaan GPU (CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--> Menggunakan device: {device}")

# Muat model YOLOv8
print("--> Memuat model YOLOv8...")
model = YOLO('weights/yolov8n.pt').to(device)
class_names = model.names
print("--> ✅ Model YOLOv8 berhasil dimuat.")

# Inisialisasi tracker OC-SORT
print("--> Menginisialisasi OC-SORT tracker...")
tracker = OCSort(det_thresh=0.4, iou_threshold=0.5, use_byte=False)
print("--> ✅ Tracker berhasil diinisialisasi.")