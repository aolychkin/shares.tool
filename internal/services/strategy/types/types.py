from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
  pass


class Deals(Base):
  __tablename__ = 'deals'
  id = Column(Integer(), primary_key=True)
  time = Column(DateTime(), nullable=False)
  transaction = Column(Float(), nullable=False)
  balance = Column(Float(), nullable=False)
  price = Column(Float(), nullable=False)
  quantity = Column(Integer(), nullable=False)
  action = Column(String(16), nullable=False)
