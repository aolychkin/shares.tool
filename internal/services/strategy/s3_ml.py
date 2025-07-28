from datetime import datetime
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
      'ADX': 'float64',  # ADX_diff - добавлю в классе
      '+DI': 'float64',
      '-DI': 'float64',
      'MACD_hist': 'float64',
      'RSI': 'float64',  # RSI_prev, RSI_diff - добавлю в классе
      'EMA24': 'float64',
      'close': 'float64',  # diff(Close, EMA24) - заменю им Close и EMA24 в классе
  }

  def __init__(self, input):
    # Инициализация БД со списком сделок текущей стратегии
    self._init_deals()

    # Обработка поступивших индикаторов
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")
    if total_nan_count > 0:
      input.dropna(inplace=True)
    self.data = self._process_input(input)

    # Обогащение данных
    self._calculate()

  def _calculate(self):
    """Вычисляет значения (обновляет self.data)."""
    self.data["ADX_diff"] = self.data['ADX'].diff()
    self.data["RSI_prev"] = self.data['RSI'].shift(1).fillna(0)
    self.data["RSI_diff"] = self.data['RSI'].diff()
    # TODO: добавить diff(Close, EMA24) - заменю им Close и EMA24 в классе

    self.data["DI_diff"] = (self.data["+DI"] - self.data["-DI"])
    self.data["ADX_power"] = self.data.apply(
        lambda row:
        row["DI_diff"] if (row["ADX"] < 26)
        else row["DI_diff"] if (row["DI_diff"] > 0)
        else row["DI_diff"],
        axis=1)

    self.data["ADX_degrees"] = self.data.apply(
        lambda row: math.degrees(math.atan2(row["ADX_diff"], 1)),
        axis=1
    )

    self.data["MACD_hist_diff"] = self.data['MACD_hist'].diff()
    self.data['MACD_hist_prev'] = self.data['MACD_hist'].shift(1)
    self.data['color_prev'] = self.data['color'].shift(1)
    self.data.dropna(inplace=True)

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
          and self.data.loc[i, 'MACD_hist_diff'] > 0.05
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

  def plot_buy(self, fig, row):
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
    ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=input[input['action'] == 'close_buy']['time'],
        y=input[input['action'] == 'close_buy']['price'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='grey'),
        name='Закрыть покупку'
    ), row=row, col=1)

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
