import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import AlertRecord

logger = logging.getLogger(__name__)


class AlertRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def insert(self, record: dict) -> AlertRecord:
        """Stage a new alert — confirmed=False by default."""
        db_record = AlertRecord(**record)
        self.db.add(db_record)
        await self.db.commit()
        await self.db.refresh(db_record)
        return db_record

    async def get_active(self) -> list[AlertRecord]:
        """
        Get all unconfirmed alerts.
        These are staged alerts awaiting human admin confirmation.
        """
        query = select(AlertRecord).where(
            AlertRecord.confirmed == False  # noqa: E712
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def confirm(self, alert_id: int) -> AlertRecord | None:
        """
        Confirm an alert — sets confirmed=True.
        Only confirmed alerts are published to the dashboard.
        """
        query = select(AlertRecord).where(AlertRecord.id == alert_id)
        result = await self.db.execute(query)
        alert = result.scalar_one_or_none()

        if alert:
            alert.confirmed = True
            await self.db.commit()
            await self.db.refresh(alert)

        return alert