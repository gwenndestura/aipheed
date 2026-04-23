from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up aiPHeed backend...")
    # Scheduler and model loading will be added in Week 14
    yield
    # Shutdown
    print("Shutting down aiPHeed backend...")


app = FastAPI(
    title="aiPHeed API",
    description="Food Insecurity Forecasting System for DSWD CALABARZON",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Welcome to aiPHeed backend!"}


app.include_router(router)