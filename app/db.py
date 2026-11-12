import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///visionguard.db")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    filename: Mapped[str] = mapped_column(String(255))
    label: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(20), default="local")


def init_db():
    Base.metadata.create_all(engine)


def add_prediction(**fields):
    with SessionLocal() as session:
        row = Prediction(**fields)
        session.add(row)
        session.commit()
        return row.id


def recent(limit=100):
    with SessionLocal() as session:
        return list(session.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)))


def ping():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
