import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.db.models import ForecastRecord, MunicipalForecastRecord

logger = logging.getLogger(__name__)

CALABARZON_PROVINCES = frozenset({
    "PH-BTG",  # Batangas
    "PH-CAV",  # Cavite
    "PH-LAG",  # Laguna
    "PH-QUE",  # Quezon
    "PH-RIZ",  # Rizal
})

DISAGGREGATION_LABEL = (
    "Spatially disaggregated estimate derived from province-level forecast "
    "using PSA 2021 SAE poverty incidence and PSA 2020 CPH population density. "
    "Municipal uncertainty is higher than province-level uncertainty."
)


class ForecastRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Province-level CRUD ───────────────────────────────────────────────

    async def insert_batch(self, records: list[dict]):
        """Save a list of province forecast records to the database."""
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

    # ── Municipal-level CRUD ──────────────────────────────────────────────

    async def insert_municipal_batch(self, records: list[dict]):
        """
        Save a list of municipal forecast records to the database.
        Automatically adds disaggregation_label if not present.
        """
        for record in records:
            if "disaggregation_label" not in record or not record["disaggregation_label"]:
                record["disaggregation_label"] = DISAGGREGATION_LABEL
            db_record = MunicipalForecastRecord(**record)
            self.db.add(db_record)
        await self.db.commit()

    async def get_municipal(
        self,
        province_code: str,
        quarter: str,
    ) -> list[MunicipalForecastRecord]:
        """
        Get all municipal forecasts for a province and quarter.
        Returns all 142 LGUs sorted by risk_index descending.
        Every row includes disaggregation_label.
        """
        query = select(MunicipalForecastRecord).where(
            MunicipalForecastRecord.province_code == province_code,
            MunicipalForecastRecord.quarter == quarter,
        ).order_by(desc(MunicipalForecastRecord.risk_index))
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_latest_municipal(self) -> list[MunicipalForecastRecord]:
        """
        Get all municipal forecasts for the most recent quarter.
        Returns all 142 LGUs across all 5 provinces.
        """
        latest_quarter_query = select(MunicipalForecastRecord.quarter).order_by(
            desc(MunicipalForecastRecord.quarter)
        ).limit(1)
        result = await self.db.execute(latest_quarter_query)
        latest_quarter = result.scalar()

        if not latest_quarter:
            return []

        query = select(MunicipalForecastRecord).where(
            MunicipalForecastRecord.quarter == latest_quarter
        ).order_by(desc(MunicipalForecastRecord.risk_index))
        result = await self.db.execute(query)
        return result.scalars().all()