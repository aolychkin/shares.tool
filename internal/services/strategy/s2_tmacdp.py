from datetime import datetime, time
import math
import sqlite3
from .types import types
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


class ByTrendMACDPower:
  """
  Стратегия входит и выходит по сигналам TrendPower
  """
  COLUMN_TYPES = {
      'close': 'float64',
      'color': 'string',
      'MACD_hist': 'float64'
  }

  def __init__(self, close, trend_power):
    self.balance = 1000
    self.storage_engine = 'sqlite:///storage/sqlite/deals_s2.db'
    self.storage = './storage/sqlite/deals_s2.db'

    engine = create_engine(self.storage_engine)
    engine.connect()
    try:
      types.Deals.__table__.drop(engine)
    except:
      print("[services.data] Table drop error")
    types.Base.metadata.create_all(engine)

    with Session(engine) as session:
      session.add(types.Deals(
          time=datetime.now(),
          transaction=self.balance,
          balance=self.balance,
          price=self.balance,
          quantity=1,
          action='income',
      ))
      session.commit()

    input = pd.concat([close, trend_power], axis=1)
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")
    if total_nan_count > 0:
      input.dropna(inplace=True)

    self.data = self._process_input(input)
    self.data["MACD_hist_diff"] = self.data['MACD_hist'].diff()
    self.data['MACD_hist_prev'] = self.data['MACD_hist'].shift(1)
    self.data['color_prev'] = self.data['color'].shift(1)
    self.data.dropna(inplace=True)

  def _process_input(self, input):
    """Приводит данные к нужным типам."""
    # Проверка колонок
    missing = set(self.COLUMN_TYPES) - set(input.columns)
    if missing:
      raise ValueError(f"[{self.__class__.__name__}] Отсутствуют колонки: {missing}")

    if str(input.index.dtype) != 'datetime64[ns]':
      raise ValueError(f"[{self.__class__.__name__}] Отсутствуют шкала времени, в качестве индекса: time")

    # Копирование и преобразование типов
    processed = input[list(self.COLUMN_TYPES.keys())].copy()
    return processed.astype(self.COLUMN_TYPES)

  # ================================================ #
  # =============== ПУБЛИЧНЫЕ МЕТОДЫ =============== #
  def simulation(self):
    """
    Если розовый (разворот на покупку) и MACD текущая гистограмма больше прошлой, то покупаем

    Если MACD текущая гисто МЕНЬШЕ прошлой, НО она зеленая, то продолжаем держать
    НО если она оранжвая (разворот на продажу) или красная (продолжение продажи), то уже продаем
    """
    open_position = False  # Флаг открытой позиции
    new_deal = NotImplemented
    for i in self.data.index:
      price = self.data.loc[i, 'close']
      # Если розовый (разворот на покупку) и MACD текущая гистограмма больше прошлой, то покупаем
      if (
          not open_position
      ) and (
          (
              # self.data.loc[i, 'color'] == 'hotpink' or
              self.data.loc[i, 'color'] == 'green'
              # and (self.data.loc[i, 'color_prev'] != 'blue' or self.data.loc[i, 'MACD_hist_prev'] > 0)
          )
          # and self.data.loc[i, 'MACD_hist_diff'] > 0.15
      ):
        open_position = True
        quantity = math.floor(self.balance / price)
        self.balance = round(self.balance-price*quantity, 2)
        new_deal = types.Deals(
            time=i,
            transaction=-price*quantity,
            balance=self.balance,
            price=price,
            quantity=quantity,
            action='buy',
        )

      # Если MACD текущая гисто МЕНЬШЕ прошлой, НО она зеленая, то продолжаем держать
      # НО если она оранжвая (разворот на продажу) или красная (продолжение продажи), то уже продаем
      elif (
          open_position
      ):
        if (
            self.data.loc[i, 'MACD_hist_diff'] <= 0
            or (
                self.data.loc[i, 'MACD_hist'] <= 0
                and self.data.loc[i, 'color'] != 'green'
                and self.data.loc[i, 'color'] != 'hotpink'
            )
        ):
          open_position = False
          self.balance = round(self.balance+price*quantity, 2)
          new_deal = types.Deals(
              time=i,
              transaction=price*quantity,
              balance=self.balance,
              price=price,
              quantity=quantity,
              action='close_buy',
          )
        # elif (
        #     i == self.data.index[-1]
        # ):
        #   pass
        elif (
            i == self.data.index[-1]
            or i.time() >= time(20, 00),
        ):
          open_position = False
          self.balance = round(self.balance+price*quantity, 2)
          new_deal = types.Deals(
              time=i,
              transaction=price*quantity,
              balance=self.balance,
              price=price,
              quantity=quantity,
              action='close_buy',
          )

      if (not new_deal == NotImplemented):
        engine = create_engine(self.storage_engine)
        engine.connect()
        with Session(engine) as session:
          session.add(new_deal)
          session.commit()

  def plot_buy(self, fig, row, col):
    """Строит график на основе результатов."""
    cnx = sqlite3.connect(self.storage)
    input = pd.read_sql_query(
        "SELECT time, price, action FROM deals",
        cnx
    )
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__} -> plot_open(): {total_nan_count}")

    # self.data['buy'] = (self.data['color'] == 'green')
    fig.add_trace(go.Scatter(
        x=input[input['action'] == 'buy']['time'],
        y=input[input['action'] == 'buy']['price'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='limegreen'),
        name='Купить'
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=input[input['action'] == 'close_buy']['time'],
        y=input[input['action'] == 'close_buy']['price'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='grey'),
        name='Закрыть покупку'
    ), row=row, col=col)

    # # Пометки сигналов покупки (зеленые стрелки вверх)
    # # self.data['sell'] = (self.data['color'] == 'red')
    # fig.add_trace(go.Scatter(
    #     x=self.data[self.data['sell']].index,
    #     y=self.data[self.data['sell']]['close'],
    #     mode='markers',
    #     marker=dict(symbol='triangle-down', size=10, color='lightpink'),
    #     name='Закрыть продажу'
    # ), row=row, col=1)

  # TODO: implement
  def close(self):
    # Пометки сигналов продажи после покупки (красные стрелки вниз)
    self.data['RSI_prev'] = self.data['RSI'].shift(1).fillna(0)
    self.data['close_buy'] = ((self.data['color'] == 'orange') & (self.data['RSI_prev'] > 70) & (self.data['RSI'] <= 70))

    # Пометки сигналов покупки после продажи (зеленые стрелки вверх)
    self.data['close_sell'] = (self.data['color'] == 'hotpink') & (self.data['RSI_prev'] < 30) & (self.data['RSI'] >= 30)

  # TODO: implement
  def plot_close(self, fig, row):
    # Пометки сигналов продажи после покупки (красные стрелки вниз)
    fig.add_trace(go.Scatter(
        x=self.data[self.data['action'] == 'close_buy'].index,
        y=self.data[self.data['action'] == 'close_buy']['price'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='grey'),
        name='Закрыть покупку'
    ), row=row, col=1)

    # # Пометки сигналов покупки после продажи (зеленые стрелки вверх)
    # fig.add_trace(go.Scatter(
    #     x=self.data[self.data['close_sell']].index,
    #     y=self.data[self.data['close_sell']]['close'],
    #     mode='markers',
    #     marker=dict(symbol='triangle-up', size=10, color='grey'),
    #     name='Закрыть продажу'
    # ), row=row, col=1)
