import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.db.models import ForecastRecord

logger = logging.getLogger(__name__)

# Canonical 5 CALABARZON province codes
CALABARZON_PROVINCES = frozenset({
    "PH-BTG",  # Batangas
    "PH-CAV",  # Cavite
    "PH-LAG",  # Laguna
    "PH-QUE",  # Quezon
    "PH-RIZ",  # Rizal
})


class ForecastRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def insert_batch(self, records: list[dict]):
        """Save a list of forecast records to the database."""
        for record in records:
            db_record = ForecastRecord(**record)
            self.db.add(db_record)
        await self.db.commit()

    async def get_latest(self) -> list[ForecastRecord]:
        """Get the most recent forecast for all 5 provinces."""
        latest_quarter_query = select(ForecastRecord.quarter).order_by(
            desc(ForecastRecord.quarter)
        ).limit(1)
        result = await self.db.execute(latest_quarter_query)
        latest_quarter = result.scalar()

        if not latest_quarter:
            return []

        query = select(ForecastRecord).where(
            ForecastRecord.quarter == latest_quarter
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_history(
        self,
        province_code: str,
        quarters: int = 8,
    ) -> list[ForecastRecord]:
        """Get forecast history for a specific province."""
        query = select(ForecastRecord).where(
            ForecastRecord.province_code == province_code
        ).order_by(desc(ForecastRecord.quarter)).limit(quarters)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_municipal(
        self,
        province_code: str,
        quarter: str,
    ) -> list[ForecastRecord]:
        """Get municipal disaggregation for a province and quarter."""
        query = select(ForecastRecord).where(
            ForecastRecord.province_code == province_code,
            ForecastRecord.quarter == quarter,
        )
        result = await self.db.execute(query)
        return result.scalars().all()