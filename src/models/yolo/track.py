from ultralytics import YOLO
import cv2
from sort import Sort  # tracker
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt")

# Inisialisasi SORT
tracker = Sort()

# Load video
cap = cv2.VideoCapture("data/raw/rekaman-mobil.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    # Ambil deteksi kendaraan (YOLO)
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label in ['car', 'truck', 'bus', 'motorbike']:  # filter hanya kendaraan
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

    # Convert ke numpy array
    dets = np.array(detections)
    tracks = tracker.update(dets)

    # Gambar tracking
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
