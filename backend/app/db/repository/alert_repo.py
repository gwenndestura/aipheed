import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import AlertRecord

logger = logging.getLogger(__name__)


class AlertRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def insert(self, record: dict) -> AlertRecord:
        """Stage a new alert — STAGED state (confirmed=False, dismissed=False)."""
        db_record = AlertRecord(**record)
        self.db.add(db_record)
        await self.db.commit()
        await self.db.refresh(db_record)
        return db_record

    async def get_staged(self) -> list[AlertRecord]:
        """Return alerts awaiting admin review (not yet confirmed or dismissed)."""
        query = select(AlertRecord).where(
            AlertRecord.confirmed == False,  # noqa: E712
            AlertRecord.dismissed == False,  # noqa: E712
        )
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_confirmed(self) -> list[AlertRecord]:
        """Return CONFIRMED alerts — published to the dashboard."""
        query = select(AlertRecord).where(AlertRecord.confirmed == True)  # noqa: E712
        result = await self.db.execute(query)
        return result.scalars().all()

    async def confirm(self, alert_id: int) -> AlertRecord | None:
        """Transition STAGED → CONFIRMED. Published to the dashboard."""
        query = select(AlertRecord).where(AlertRecord.id == alert_id)
        result = await self.db.execute(query)
        alert = result.scalar_one_or_none()

        if alert and not alert.dismissed:
            alert.confirmed = True
            await self.db.commit()
            await self.db.refresh(alert)

        return alert

    async def dismiss(self, alert_id: int) -> AlertRecord | None:
        """Transition STAGED → DISMISSED. Kept for audit only, not shown publicly."""
        query = select(AlertRecord).where(AlertRecord.id == alert_id)
        result = await self.db.execute(query)
        alert = result.scalar_one_or_none()

        if alert and not alert.confirmed:
            alert.dismissed = True
            await self.db.commit()
            await self.db.refresh(alert)

        return alert