from fastapi import FastAPI

from backend.api.auth import router as auth_router


app = FastAPI(
    title="Fire Detection Backend",
    description="Fire source classification API",
    version="1.0.0",
)


app.include_router(auth_router)
