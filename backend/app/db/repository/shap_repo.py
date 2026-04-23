import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.db.models import SHAPRecord

logger = logging.getLogger(__name__)


class SHAPRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def insert_batch(self, records: list[dict]):
        """Save a list of SHAP records to the database."""
        for record in records:
            db_record = SHAPRecord(**record)
            self.db.add(db_record)
        await self.db.commit()

    async def get_by_province_quarter(
        self,
        province_code: str,
        quarter: str,
    ) -> list[SHAPRecord]:
        """
        Get all SHAP feature values for a specific province and quarter.
        Returns features sorted by mean_abs_shap descending.
        """
        query = select(SHAPRecord).where(
            SHAPRecord.province_code == province_code,
            SHAPRecord.quarter == quarter,
        ).order_by(desc(SHAPRecord.mean_abs_shap))
        result = await self.db.execute(query)
        return result.scalars().all()