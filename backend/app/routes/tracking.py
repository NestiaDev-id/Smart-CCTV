# backend/app/routes/tracking.py

import asyncio
import cv2
import base64
import numpy as np
import yt_dlp
import socketio
from ..core.model import model, tracker, class_names # Impor model dari core

# Inisialisasi Socket.IO Server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# Dictionary untuk menyimpan tugas pemrosesan video untuk setiap koneksi
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

            results = model(frame, stream=False, verbose=False)
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    detections.append([x1, y1, x2, y2, conf, cls])
            
            detections_np = np.array(detections)

            if len(detections_np) > 0:
                tracked_objects = tracker.update(detections_np, frame)
                for obj in tracked_objects:
                    x1, y1, x2, y2, obj_id, cls_id, conf = obj
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"ID: {int(obj_id)} {class_names[int(cls_id)]}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
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
    # Validasi data menggunakan skema Pydantic
    # from ..schemas.tracking import YouTubeUrlPayload
    # try:
    #     payload = YouTubeUrlPayload(**data)
    #     url = str(payload.url)
    # except Exception:
    #     await sio.emit('error', {'message': 'URL tidak valid.'}, to=sid)
    #     return
    
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