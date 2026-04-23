from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@router.get("/readiness")
async def readiness_check():
    # DB check will be added in Week 11
    return {"status": "ready"}