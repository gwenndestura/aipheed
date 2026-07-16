"""
app/db/repository/shap_repo.py
---------------------------------
Repository for SHAPRecord and DriverRecord DB operations.

SHAPRepository  → feature-level SHAP (45 rows per province-quarter)
DriverRepository → grouped driver SHAP (5 rows per province-quarter)
"""

import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.db.models import SHAPRecord, DriverRecord

logger = logging.getLogger(__name__)


class SHAPRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def insert_batch(self, records: list[dict]) -> None:
        """Save SHAP records for one or more province-quarters."""
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
        Get all 45 SHAP feature records for a province-quarter.
        Returned sorted by |shap_value| descending (highest-impact first).
        """
        query = (
            select(SHAPRecord)
            .where(
                SHAPRecord.province_code == province_code,
                SHAPRecord.quarter == quarter,
            )
            .order_by(desc(SHAPRecord.mean_abs_shap))
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_latest_by_province(
        self,
        province_code: str,
    ) -> list[SHAPRecord]:
        """Get SHAP records for the most recent quarter of a province."""
        latest_q_query = (
            select(SHAPRecord.quarter)
            .where(SHAPRecord.province_code == province_code)
            .order_by(desc(SHAPRecord.quarter))
            .limit(1)
        )
        result = await self.db.execute(latest_q_query)
        latest_q = result.scalar()
        if not latest_q:
            return []
        return await self.get_by_province_quarter(province_code, latest_q)


class DriverRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def insert_batch(self, records: list[dict]) -> None:
        """Save driver group records for one or more province-quarters."""
        for record in records:
            db_record = DriverRecord(**record)
            self.db.add(db_record)
        await self.db.commit()

    async def get_by_province_quarter(
        self,
        province_code: str,
        quarter: str,
    ) -> list[DriverRecord]:
        """
        Get all 5 driver records for a province-quarter.
        Returned sorted by |group_shap| descending (highest-impact first).
        """
        query = (
            select(DriverRecord)
            .where(
                DriverRecord.province_code == province_code,
                DriverRecord.quarter == quarter,
            )
            .order_by(desc(DriverRecord.display_pct))
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_latest_by_province(
        self,
        province_code: str,
    ) -> list[DriverRecord]:
        """Get driver records for the most recent quarter of a province."""
        latest_q_query = (
            select(DriverRecord.quarter)
            .where(DriverRecord.province_code == province_code)
            .order_by(desc(DriverRecord.quarter))
            .limit(1)
        )
        result = await self.db.execute(latest_q_query)
        latest_q = result.scalar()
        if not latest_q:
            return []
        return await self.get_by_province_quarter(province_code, latest_q)
