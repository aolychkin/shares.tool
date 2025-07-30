from datetime import datetime
from datetime import time
import math
import sqlite3
from .types import types
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


# TODO: передать в нейронку: ADX, +DI, -DI, ADX_diff, diff(Close, EMA24), RSI, RSI_prev, RSI_diff, MACD_hist
class ML:
  """
  Стратегия входит и выходит по сигналам ML
  Требуемые показатели: 'ADX', '+DI', '-DI', 'MACD_hist', 'RSI', 'EMA24', 'close'
  """
  COLUMN_TYPES = {
      'close': 'float64',
      'high': 'float64',
      # Candles
      'change': 'float64',
      'a_low': 'float64',
      'a_high': 'float64',
      'tail_low': 'float64',
      'tail_high': 'float64',
      'change_major': 'float64',
      'a_low_major': 'float64',
      'a_high_major': 'float64',
      'tail_low_major': 'float64',
      'tail_high_major': 'float64',
      # ADX
      'ADX': 'float64',
      '+DI': 'float64',
      '-DI': 'float64',
      'ADX_major': 'float64',
      '+DI_major': 'float64',
      '-DI_major': 'float64',
      # EMA
      'a_EMA55': 'float64',
      'a_EMA24': 'float64',
      'a_EMA9': 'float64',
      'a_EMA55_major': 'float64',
      'a_EMA24_major': 'float64',
      'a_EMA9_major': 'float64',
      # MACD
      'MACD': 'float64',
      'MACD_signal': 'float64',
      'MACD_hist': 'float64',
      'MACD_major': 'float64',
      'MACD_signal_major': 'float64',
      'MACD_hist_major': 'float64',
      # RSI
      'RSI': 'float64',
      'RSI_major': 'float64',
      # Trend power
      'ADX_degrees': 'float64',
      'ADX_degrees_major': 'float64',
      # Предсказание и уверенность в себе от AI
      'prediction': 'string',
      'confidence': 'float64',
  }

  # TODO: вынести в конфиг
  # Надо синхронизировать с классом Analysis оба параметра (CATEGORIES, CONFIDENCE)
  CATEGORIES = ['Падение (< -0.2%)', 'Стабильность (-0.2% до 0.2%)', 'Рост (> 0.2%)']
  CONFIDENCE = 0.85  # 0.88
  PERCENT = 0.2
  FEE = 0.05

  def __init__(self, input):
    # Обработка поступивших индикаторов
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")
    if total_nan_count > 0:
      input.dropna(inplace=True)
    self.data = self._process_input(input)

    # Инициализация БД со списком сделок текущей стратегии
    self._init_deals()

  def _init_deals(self):
    self.balance = 1000
    self.storage_engine = 'sqlite:///storage/sqlite/deals_s3.db'
    self.storage = './storage/sqlite/deals_s3.db'

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
          confidence=1,
      ))
      session.commit()

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

  def _delta(self, old, new):
    return ((new - old)/old*100)
  # ================================================ #
  # =============== ПУБЛИЧНЫЕ МЕТОДЫ =============== #

  def simulation(self):
    open_position = False  # Флаг открытой позиции
    new_deal = NotImplemented
    last_buy_price = 0

    for i in self.data.index:
      price_buy = self.data.loc[i, 'close']
      price_sell = self.data.loc[i, 'high']
      # Если розовый (разворот на покупку) и MACD текущая гистограмма больше прошлой, то покупаем
      if (
          not open_position
      ) and (
          self.data.loc[i, 'prediction'] == self.CATEGORIES[2]
          and self.data.loc[i, 'confidence'] >= self.CONFIDENCE
      ):
        open_position = True
        last_buy_price = price_buy
        price_buy = price_buy * (1 + self.FEE/100)
        quantity = math.floor(self.balance / price_buy)
        self.balance = round(self.balance-price_buy*quantity, 2)
        new_deal = types.Deals(
            time=i,
            transaction=-price_buy*quantity,
            balance=self.balance,
            price=price_buy,
            quantity=quantity,
            action='buy',
            confidence=self.data.loc[i, 'confidence'],
        )
      elif (
          open_position and last_buy_price != 0
      ):
        if (
            self._delta(old=last_buy_price, new=price_sell) >= self.PERCENT
        ):
          price_sell = last_buy_price * (1 + self.PERCENT/100)
          price_sell = price_sell * (1 - self.FEE/100)
          open_position = False
          self.balance = round(self.balance+price_sell*quantity, 2)
          new_deal = types.Deals(
              time=i,
              transaction=price_sell*quantity,
              balance=self.balance,
              price=price_sell,
              quantity=quantity,
              action='close_buy',
              confidence=self.data.loc[i, 'confidence'],
          )
        elif (
            self._delta(old=last_buy_price, new=price_buy) <= -self.PERCENT
            and self._delta(old=last_buy_price, new=price_sell) > 0
        ):
          price_sell = last_buy_price
          price_sell = price_sell * (1 - self.FEE/100)
          open_position = False
          self.balance = round(self.balance+price_sell*quantity, 2)
          new_deal = types.Deals(
              time=i,
              transaction=price_sell*quantity,
              balance=self.balance,
              price=price_sell,
              quantity=quantity,
              action='close_buy',
              confidence=self.data.loc[i, 'confidence'],
          )
        elif (
            self._delta(old=last_buy_price, new=price_buy) <= -self.PERCENT * 2
        ):
          price_sell = last_buy_price * (1 - (self.PERCENT * 2)/100)
          price_sell = price_sell * (1 - self.FEE/100)
          open_position = False
          self.balance = round(self.balance+price_sell*quantity, 2)
          new_deal = types.Deals(
              time=i,
              transaction=price_sell*quantity,
              balance=self.balance,
              price=price_sell,
              quantity=quantity,
              action='close_buy',
              confidence=self.data.loc[i, 'confidence'],
          )
        elif (
            time(20, 30) <= i.time()
            # not (time(7, 0) <= i.time() <= time(15, 40))
        ):
          price_sell = price_buy
          price_sell = price_sell * (1 - self.FEE/100)
          open_position = False
          self.balance = round(self.balance+price_sell*quantity, 2)
          new_deal = types.Deals(
              time=i,
              transaction=price_sell*quantity,
              balance=self.balance,
              price=price_sell,
              quantity=quantity,
              action='close_buy',
              confidence=self.data.loc[i, 'confidence'],
          )

      if (not new_deal == NotImplemented):
        engine = create_engine(self.storage_engine)
        engine.connect()
        with Session(engine) as session:
          session.add(new_deal)
          session.commit()

  def plot_buy(self, fig, row=1, col=1):
    """Строит график на основе результатов."""
    cnx = sqlite3.connect(self.storage)
    input = pd.read_sql_query(
        "SELECT time, price, action, confidence FROM deals",
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
        text=input[input['action'] == 'buy']['confidence'],
        name='Купить'
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=input[input['action'] == 'close_buy']['time'],
        y=input[input['action'] == 'close_buy']['price'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='grey'),
        text=input[input['action'] == 'close_buy']['confidence'],
        name='Закрыть покупку'
    ), row=row, col=col)

    return fig

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
