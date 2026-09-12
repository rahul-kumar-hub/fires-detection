from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.auth import router as auth_router
from backend.api.predictions import router as predictions_router


app = FastAPI(
    title="Fire Detection API",
    description="Fire source classification and authentication API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "success": True,
        "message": "Fire Detection API is running",
    }


app.include_router(auth_router)
app.include_router(predictions_router)
