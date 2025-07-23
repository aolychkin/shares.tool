from datetime import datetime
from datetime import date
import time
from calendar import monthrange
from tqdm import tqdm
from tabulate import tabulate

import pandas as pd

from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from internal.domain.types import types
from internal.config import config


def load_day_candle(m, d):
  FIGI = 'BBG004730RP0'
  data = []
  data_candle = []

  with Client(config.TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
    for day in tqdm(range(d-3, d+1)):
      for candle in client.market_data.get_candles(
              figi=FIGI,
              from_=datetime(2025, m, day, 6, 0),
              to=datetime(2025, m, day, 23, 59),
              interval=CandleInterval.CANDLE_INTERVAL_2_MIN).candles:
        if day == d and datetime(2024, 12, 2, 7, 00).strftime("%H:%M") < candle.time.strftime("%H:%M") <= datetime(2024, 12, 2, 15, 30).strftime("%H:%M"):
          data.append([
              str(candle.time),
              candle.open.units + candle.open.nano / 1000000000,
              candle.high.units + candle.high.nano / 1000000000,
              candle.low.units + candle.low.nano / 1000000000,
              candle.close.units + candle.close.nano / 1000000000,
              candle.volume,
          ])
          data_candle.append([
              str(candle.time),
              candle.open.units + candle.open.nano / 1000000000,
              candle.high.units + candle.high.nano / 1000000000,
              candle.low.units + candle.low.nano / 1000000000,
              candle.close.units + candle.close.nano / 1000000000,
              candle.volume,
          ])
        else:
          data.append([
              str(candle.time),
              candle.open.units + candle.open.nano / 1000000000,
              candle.high.units + candle.high.nano / 1000000000,
              candle.low.units + candle.low.nano / 1000000000,
              candle.close.units + candle.close.nano / 1000000000,
              candle.volume,
          ])

  df = pd.DataFrame(data)
  df.columns = ["time",  "open", "high", "low", "close",  "volume"]
  df.insert(loc=0, column='id', value=df.index)
  df_candle = pd.DataFrame(data_candle)
  df_candle.columns = ["time",  "open", "high", "low", "close",  "volume"]
  df_candle.insert(loc=0, column='id', value=df_candle.index+df.loc[df['time'] == f"2025-01-{d} 07:02:00+00:00"].index.values[0])
  # print(tabulate(df.loc[1050:1070], headers='keys', tablefmt='psql'))
  # print(tabulate(df_candle.loc[:20], headers='keys', tablefmt='psql'))

  return df, df.loc[df['time'] == f"2025-01-{d} 07:02:00+00:00"].index.values[0], df.loc[df['time'] == f"2025-01-{d} 15:30:00+00:00"].index.values[0]


def load_candles(years, engine):
  FIGI = 'BBG004730RP0'

  with Session(engine) as session:
    with Client(config.TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
      for year in years:
        print(f"Start to getting year: {year}")
        for _, month in tqdm(enumerate(range(1, 13))):
          for day in tqdm(range(monthrange(year, month)[1])):
            for candle in client.market_data.get_candles(
                    figi=FIGI,
                    from_=datetime(year, month, day+1, 6, 0),
                    to=datetime(year, month, day+1, 23, 59),
                    interval=CandleInterval.CANDLE_INTERVAL_3_MIN).candles:
              data = types.Candles(
                  figi=FIGI,
                  open=candle.open.units + candle.open.nano / 1000000000,
                  high=candle.high.units + candle.high.nano / 1000000000,
                  low=candle.low.units + candle.low.nano / 1000000000,
                  close=candle.close.units + candle.close.nano / 1000000000,
                  volume=candle.volume,
                  time=candle.time,
                  is_complete=candle.is_complete
              )
              session.add(data)
          time.sleep(1)
      session.commit()


if __name__ == '__main__':
  # load_day_candle(1, 16)
  engine = create_engine('sqlite:///storage/sqlite/shares.db')
  engine.connect()
  try:
    types.Candles.__table__.drop(engine)
  except:
    print("[services.data] Table drop error")
  types.Base.metadata.create_all(engine)
  # 2017
  load_candles(range(2025, 2026), engine)
