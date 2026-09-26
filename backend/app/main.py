from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.router import api_router


app = FastAPI(
    title="Dermatology AI Agents",
    description="AI-assisted dermatology image analysis system",
    version="0.1.0",
)

app.mount(
    "/gradcam",
    StaticFiles(directory="models/gradcam"),
    name="gradcam",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "message": "Dermatology AI Agents API",
        "status": "running",
        "version": "0.1.0",
    }