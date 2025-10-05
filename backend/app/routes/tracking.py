# backend/app/routes/tracking.py

import asyncio
import cv2
import base64
import numpy as np
import yt_dlp
import socketio
from ..core.model import model, tracker, class_names # Impor model & tracker dari core

# Inisialisasi Socket.IO Server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# Dictionary untuk menyimpan tugas pemrosesan video
video_processing_tasks = {}

def get_video_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[ext=webm][height<=720]',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict.get('url', None)

async def process_video(sid, stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        await sio.emit('error', {'message': 'Gagal membuka stream video.'}, to=sid)
        return

    try:
        while sid in video_processing_tasks:
            ret, frame = cap.read()
            if not ret:
                break
            
            # algoritma tracking steps

            # 1. Deteksi objek dengan YOLOv8
            results = model(frame, stream=False, verbose=False)
            
            # 2. Siapkan data deteksi untuk DeepSORT
            # Format: [ [x1, y1, w, h], confidence, class_id ]
            detections_for_deepsort = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Konversi (x1, y1, x2, y2) ke (x, y, w, h)
                    w = x2 - x1
                    h = y2 - y1
                    detections_for_deepsort.append(
                        ([int(x1), int(y1), int(w), int(h)], conf, class_names[cls])
                    )

            # 3. Update tracker DeepSORT
            # Parameter kedua adalah frame, yang digunakan DeepSORT untuk fitur penampilan
            tracks = tracker.update_tracks(detections_for_deepsort, frame=frame)

            # 4. Gambar bounding box dan ID dari hasil tracking
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb() # Dapatkan format [left, top, right, bottom]
                
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                class_name = track.get_det_class()
                
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track_id} {class_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Encode frame ke format JPEG lalu ke Base64 untuk dikirim
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Kirim frame ke client
            await sio.emit('video_frame', {'image': frame_base64}, to=sid)
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"Error saat memproses video: {e}")
        await sio.emit('error', {'message': str(e)}, to=sid)
    finally:
        cap.release()
        print(f"Stream untuk SID {sid} telah ditutup.")
        if sid in video_processing_tasks:
            del video_processing_tasks[sid]


@sio.on('connect')
def connect(sid, environ):
    print(f"🔗 Client terhubung: {sid}")

@sio.on('disconnect')
def disconnect(sid):
    print(f"🔌 Client terputus: {sid}")
    if sid in video_processing_tasks:
        video_processing_tasks[sid].cancel()
        del video_processing_tasks[sid]
        print(f"Tugas pemrosesan video untuk SID {sid} dihentikan.")

@sio.on('process_youtube_url')
async def process_youtube_url(sid, data):
    url = data.get('url')
    if not url:
        await sio.emit('error', {'message': 'URL tidak ditemukan.'}, to=sid)
        return
        
    print(f"Menerima URL YouTube dari {sid}: {url}")
    
    if sid in video_processing_tasks:
        video_processing_tasks[sid].cancel()

    stream_url = get_video_stream_url(url)
    if not stream_url:
        await sio.emit('error', {'message': 'Gagal mendapatkan stream URL dari YouTube.'}, to=sid)
        return

    task = asyncio.create_task(process_video(sid, stream_url))
    video_processing_tasks[sid] = task
    await sio.emit('processing_started', {'message': 'Pemrosesan video dimulai.'})