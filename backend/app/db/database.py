from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

# echo was hardcoded on, which logs every statement -- including the email on
# each login lookup -- to stdout in production. Off by default; set DB_ECHO=1
# to get it back while debugging.
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass