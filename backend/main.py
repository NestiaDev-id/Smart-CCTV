import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
from app.routes.tracking import sio

app = FastAPI(
    title="Smart CCTV Backend",
    description="Backend untuk live object tracking dari stream YouTube.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

socket_app = socketio.ASGIApp(sio)
app.mount("/ws", socket_app)

@app.get("/")
async def root():
    return {"message": "Server backend is running"}

@app.get("/status")
def read_status():
    return {"status": "ok", "message": "Server backend Smart CCTV aktif!"}

# if __name__ == "__main__":
#     print("🚀 Menjalankan server backend pada http://0.0.0.0:8000")
#     uvicorn.run(
#         "main:app",
#         port=8000,
#         reload=True
#     )