from fastapi import FastAPI

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Smart CCTV API",
    description="API untuk mengelola dan memproses data dari Smart CCTV",
    version="1.0.0"
)

@app.get("/", tags=["Root"])
def read_root():
    """
    Endpoint utama untuk mengecek apakah API berjalan.
    """
    return {"message": "Backend is running!"}