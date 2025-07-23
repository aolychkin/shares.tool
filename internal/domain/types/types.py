from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
  pass

class Candles(Base):
  __tablename__ = 'candles'
  id = Column(Integer(), primary_key=True)
  figi = Column(String(16), nullable=False)
  time = Column(DateTime(), nullable=False)
  open = Column(Float(), nullable=False)
  high = Column(Float(), nullable=False)
  low = Column(Float(), nullable=False)
  close = Column(Float(), nullable=False)
  volume = Column(Integer(), nullable=False)
  is_complete = Column(Boolean(), default=0, nullable=False)
