from fastapi import APIRouter

from app.api.v1 import admin, auth, dashboard, forecast, health, insights, report

router = APIRouter()

# The dashboard contract lives under /api/v1, the base path the frontend spec
# fixes. Health keeps its original /v1 mount so existing probes keep working.
router.include_router(health.router, prefix="/v1", tags=["Health"])

API_V1 = "/api/v1"
router.include_router(health.router, prefix=API_V1, tags=["Health"])
router.include_router(dashboard.router, prefix=API_V1, tags=["Dashboard"])
router.include_router(forecast.router, prefix=API_V1, tags=["Forecast"])
router.include_router(insights.router, prefix=API_V1, tags=["Insights"])
router.include_router(report.router, prefix=API_V1, tags=["Report"])

# Auth, and the public face of the review pipeline (rejections, feedback
# submission) -- both unauthenticated by design.
router.include_router(auth.router, prefix=f"{API_V1}/auth", tags=["Auth"])
router.include_router(admin.public_router, prefix=API_V1, tags=["Publication"])

# Everything that changes publication state. The router carries the admin
# dependency, so a new route added here is gated by default rather than by
# remembering to add it.
router.include_router(admin.admin_router, prefix=f"{API_V1}/admin", tags=["Admin"])
