from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import os

# Inisialisasi aplikasi
app = FastAPI(
    title="API Prediksi Kategori Capaian RKT",
    description="Menentukan kategori capaian realisasi RKT PBPH berdasarkan input fitur",
    version="1.0.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model secara aman
model_path = "model_rf.pkl"
model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")

# Definisi input
class InputData(BaseModel):
    Tahun: int = Field(..., example=2024)
    Jenis_Hutan: int = Field(..., example=1)  # 0=Hutan Alam, 1=Hutan Tanaman
    Luas_PBPH: float = Field(..., example=15000)
    Target_Murni_Luas: float = Field(..., example=950)
    Target_Murni_Volume: float = Field(..., example=60000)
    Target_Carry_Luas: float = Field(..., example=700)
    Target_Carry_Volume: float = Field(..., example=20000)
    Target_Jumlah_Luas: float = Field(..., example=1000)
    Realisasi_Alam: float = Field(..., example=30000)
    Realisasi_Tanaman: float = Field(..., example=5000)
    log_Realisasi_Produksi: float = Field(..., example=10.2)
    Rasio_Realisasi: float = Field(..., example=0.7)
    Skala_PBPH: float = Field(..., example=2e8)

# Endpoint prediksi
@app.post("/predict", summary="Prediksi Kategori Capaian", tags=["Prediction"])
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model tidak tersedia.")

    try:
        # Ubah ke array sesuai urutan fitur training
        input_array = np.array([[  
            data.Tahun,
            data.Jenis_Hutan,
            data.Luas_PBPH,
            data.Target_Murni_Luas,
            data.Target_Murni_Volume,
            data.Target_Carry_Luas,
            data.Target_Carry_Volume,
            data.Target_Jumlah_Luas,
            data.Realisasi_Alam,
            data.Realisasi_Tanaman,
            data.log_Realisasi_Produksi,
            data.Rasio_Realisasi,
            data.Skala_PBPH
        ]])

        # Prediksi menggunakan model
        prediction = model.predict(input_array)[0]

        # Mapping hasil prediksi ke label capaian
        label_mapping = {
            0: "0% (Tidak Ada Realisasi)",
            1: "1–50%",
            2: "51–100%",
            3: ">100%"
        }

        return {
            "Kategori": int(prediction),
            "Capaian": label_mapping.get(int(prediction), "Tidak diketahui")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {str(e)}")
