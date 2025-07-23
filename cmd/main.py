from datetime import datetime
import math
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# TODO: Добавить: 1. Линию силы тренда. Линию покупки и продажи. Точку предлагаемого действия


def draw_figure(df, indicators, deals):
  row_heights = [0.5]
  for _ in range(len(indicators)):
    row_heights.append(0.5/len(indicators))

  fig = make_subplots(rows=len(indicators)+1, cols=1, shared_xaxes=True,
                      vertical_spacing=0.03,
                      row_heights=row_heights)
  # Добавление графика цены
  add_candles(fig, df, 1)
  add_EMA(fig, df, 1)
  add_RSI(fig, df, 3)
  add_ADX(fig, df, 4)
  add_MACD(fig, df, 5)
  add_TrendPower(fig, df, 2)
  add_CloseDeal(fig, df, 1, deals)
  add_OpenDeal(fig, df, 1, deals)

  # Настройка макета
  fig.update_layout(
      title_text="Цена и RSI",
      yaxis_title="Цена",
      yaxis2_title="Trend Power",
      yaxis3_title="RSI",
      yaxis4_title="ADX",
      yaxis5_title="MACD",
      xaxis_rangeslider_visible=False,
      height=1300,
      # yaxis_range=[min(df['low']), max(df['high'])],
      template="plotly_dark"  # Или другой шаблон по вашему вкусу
  )

  # Отображение графика
  fig.show()


# Торговый алгоритм с ограничением на одну открытую сделку
def add_OpenDeal(fig, df, row, deals):
  open_position = False  # Флаг открытой позиции
  df['buy'] = False
  df['sell'] = False
  for i in range(len(df)):
    time = df.loc[i, 'time']
    price = df.loc[i, 'close']

    # TODO: Добавить, что это торговый день И (мб) основная сессия = убрать все периоды аля флета (где 1-2 сделки)
    # TODO: Научиться покупать на красном, если не было раньше сделок о покупке (мб)
    # Если нет открытой позиции и есть сигнал на покупку
    if (not open_position) and (df.loc[i, 'color'] == 'green' and df.loc[i, 'ADX_degrees'] >= 25 and df.loc[i, 'ADX_degrees'] < 80):
      df.loc[i, 'buy'] = True
      balance = deals.loc[len(deals)-1, 'balance']
      quantity = math.floor(deals.loc[len(deals)-1, 'balance'] / price)
      deals.loc[len(deals)] = [
          time, -price*quantity, round(balance-price*quantity), price, quantity, 0, 'buy'
      ]
      open_position = True
      # columns=['time', 'transaction', 'balance', 'price', 'quantity', 'is_closed', 'cause'])

    # Если есть открытая позиция и есть сигнал на продажу
    elif open_position and (df.loc[i, 'close_buy'] or i == len(df)-1):
      df.loc[i, 'sell'] = True
      balance = deals.loc[len(deals)-1, 'balance']
      quantity = deals.loc[len(deals)-1, 'quantity']
      deals.loc[len(deals)] = [
          time, price*quantity, balance+price*quantity, price, quantity, 0, 'close_buy'
      ]
      open_position = False

  # df['buy'] = (df['color'] == 'green')
  fig.add_trace(go.Scatter(
      x=df[df['buy']]['time'],
      y=df[df['buy']]['close'],
      mode='markers',
      marker=dict(symbol='triangle-up', size=10, color='limegreen'),
      name='Купить'
  ), row=row, col=1)

  # Пометки сигналов покупки (зеленые стрелки вверх)
  # df['sell'] = (df['color'] == 'red')
  fig.add_trace(go.Scatter(
      x=df[df['sell']]['time'],
      y=df[df['sell']]['close'],
      mode='markers',
      marker=dict(symbol='triangle-down', size=10, color='lightpink'),
      name='Закрыть продажу'
  ), row=row, col=1)


def add_CloseDeal(fig, df, row, deals):
  # Пометки сигналов продажи (красные стрелки вниз)
  df['RSI_prev'] = df['RSI'].shift(1).fillna(0)
  df['close_buy'] = ((df['color'] == 'orange') & (df['RSI_prev'] > 70) & (df['RSI'] <= 70))
  fig.add_trace(go.Scatter(
      x=df[df['close_buy']]['time'],
      y=df[df['close_buy']]['close'],
      mode='markers',
      marker=dict(symbol='triangle-down', size=10, color='grey'),
      name='Закрыть покупку'
  ), row=row, col=1)

  # Пометки сигналов покупки (зеленые стрелки вверх)
  df['close_sell'] = (df['color'] == 'hotpink') & (df['RSI_prev'] < 30) & (df['RSI'] >= 30)
  fig.add_trace(go.Scatter(
      x=df[df['close_sell']]['time'],
      y=df[df['close_sell']]['close'],
      mode='markers',
      marker=dict(symbol='triangle-up', size=10, color='grey'),
      name='Закрыть продажу'
  ), row=row, col=1)


# Для ADX все что не рост - то слабость тренда
def add_TrendPower(fig, df, row):

  df["ADX_diff"] = df['ADX'].diff()
  df["color"] = df.apply(
      lambda row:
      'grey' if (row["ADX"] < 25)
      else 'green' if (row["+DI"] > row["-DI"] and row["ADX_diff"] > 0 and row['close'] > row['EMA24'])
      else 'darkgreen' if (row["+DI"] > row["-DI"] and row["ADX_diff"] > 0)
      else 'orange' if (row["+DI"] > row["-DI"])
      else 'red' if (row["+DI"] < row["-DI"] and row["ADX_diff"] > 0 and row['close'] < row['EMA24'])
      else 'darkred' if (row["+DI"] < row["-DI"] and row["ADX_diff"] > 0)
      else 'hotpink',
      axis=1
  )

  df["DI_diff"] = (df["+DI"] - df["-DI"])
  df["adx_power"] = df.apply(
      lambda row:
      row["DI_diff"] if (row["ADX"] < 25)
      else row["DI_diff"] + row['ADX'] if (row["DI_diff"] > 0)
      else row["DI_diff"] - row['ADX'],
      axis=1)

  df["ADX_degrees"] = df.apply(
      lambda row: math.degrees(math.atan2(row["ADX_diff"], 1)),
      axis=1
  )

  fig.add_trace(go.Bar(x=df['time'], y=df['adx_power'], name='Trend Power', marker_color=df["color"]), row=row, col=1)
  fig.add_trace(go.Scatter(x=df['time'], y=df['ADX_degrees'], mode='lines', name='ADX_degrees', marker_color='blue'), row=row, col=1)


# # Для ADX все что не рост - то слабость тренда
# def add_TrendPower(fig, df, row):
#   df["DI_diff"] = (df["+DI"] - df["-DI"])
#   df["color"] = df.apply(lambda row: 'green' if (row["+DI"] > row["-DI"]) else 'red', axis=1)
#   df["adx_power"] = df.apply(
#       lambda row:
#       row["DI_diff"] if (row["ADX"] < 25)
#       else row["DI_diff"] + row['ADX'] if (row["DI_diff"] > 0)
#       else row["DI_diff"] - row['ADX'],
#       axis=1)

#   fig.add_trace(go.Bar(x=df['time'], y=df['adx_power'], name='Trend Power', marker_color=df["color"]), row=row, col=1)
#   # fig.add_trace(go.Scatter(x=df['time'], y=df['DI_diff'], mode='lines', name='+DI', marker_color='yellow'), row=row, col=1)
#   # fig.add_trace(go.Scatter(x=df['time'], y=[1] * len(df.index), mode='lines', name='Overbought', line=dict(color='grey', dash='dash'), showlegend=False), row=row, col=1)


def add_EMA(fig, df, row):
  df["EMA55"] = EMAIndicator(close=df["close"], window=55, fillna=False).ema_indicator()
  df["EMA24"] = EMAIndicator(close=df["close"], window=24, fillna=False).ema_indicator()
  df["EMA9"] = EMAIndicator(close=df["close"], window=9, fillna=False).ema_indicator()

  fig.add_trace(go.Scatter(x=df['time'], y=df['EMA55'], mode='lines', name='EMA55', marker_color='yellow'), row=row, col=1)
  fig.add_trace(go.Scatter(x=df['time'], y=df['EMA24'], mode='lines', name='EMA24', marker_color='blue'), row=row, col=1)
  fig.add_trace(go.Scatter(x=df['time'], y=df['EMA9'], mode='lines', name='EMA9', marker_color='red'), row=row, col=1)


def add_MACD(fig, df, row):
  ta_MACD = MACD(close=df["close"], window_fast=12, window_slow=24, window_sign=10, fillna=False)
  df["MACD_signal"] = ta_MACD.macd_signal()
  df["MACD"] = ta_MACD.macd()

  fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], mode='lines', name='MACD', marker_color='blue'), row=row, col=1)
  fig.add_trace(go.Scatter(x=df['time'], y=df['MACD_signal'], mode='lines', name='MACD_signal', marker_color='red'), row=row, col=1)


def add_ADX(fig, df, row):
  ta_ADX = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=9, fillna=False)
  df["ADX"] = ta_ADX.adx()
  df["+DI"] = ta_ADX.adx_pos()
  df["-DI"] = ta_ADX.adx_neg()

  fig.add_trace(go.Scatter(x=df['time'], y=df['ADX'], mode='lines', name='ADX', marker_color='orange'), row=row, col=1)
  fig.add_trace(go.Scatter(x=df['time'], y=df['+DI'], mode='lines', name='+DI', marker_color='green'), row=row, col=1)
  fig.add_trace(go.Scatter(x=df['time'], y=df['-DI'], mode='lines', name='-DI', marker_color='red'), row=row, col=1)
  fig.add_trace(go.Scatter(x=df['time'], y=[25] * len(df.index), mode='lines', name='Overbought', line=dict(color='grey', dash='dash'), showlegend=False), row=row, col=1)


# Сигналом является только выход из зоны 70/30, вход и нахождение в зоне сигналом НЕ является
# Нам интересна только та свеча, на которой возник сигнал 70/30. Если АДХ положительный и растущий, то ничего не делаем при этом
def add_RSI(fig, df, row):
  # Расчет показателя RSI
  df['RSI'] = RSIIndicator(close=df['close'], window=9, fillna=False).rsi()
  fig.add_trace(go.Scatter(x=df['time'], y=df['RSI'], mode='lines', name='RSI', marker_color='orange'), row=row, col=1)
  # Добавление уровней перекупленности/перепроданности
  fig.add_trace(go.Scatter(x=df['time'], y=[70] * len(df.index), mode='lines', name='Overbought', line=dict(color='red', dash='dash'), showlegend=False), row=row, col=1)
  fig.add_trace(go.Scatter(x=df['time'], y=[30] * len(df.index), mode='lines', name='Oversold', line=dict(color='green', dash='dash'), showlegend=False), row=row, col=1)


def add_candles(fig, df, row):
  fig.add_trace(go.Candlestick(
      x=df['time'],
      open=df['open'],
      high=df['high'],
      low=df['low'],
      close=df['close'],
      text=round((df["close"]/df["close"].shift(1)-1)*100, 2),
      name='Chandlestick'),
      row=row,
      col=1
  )


if __name__ == '__main__':
  cnx = sqlite3.connect('./storage/sqlite/shares.db')
  candles = pd.read_sql_query(
      # "SELECT id, time, open, high, low, close, volume FROM candles WHERE time >= '2025-06-01 06:00:00.000' and time <= '2025-06-30 23:59:00.00'", cnx)
      "SELECT id, time, open, high, low, close, volume FROM candles WHERE time >= '2025-07-10 06:00:00.000' and time <= '2025-07-10 23:00:00.00'", cnx)
  # "SELECT id, time, open, high, low, close, volume FROM candles WHERE time >= '2025-07-14 06:00:00.000' and time <= '2025-07-14 15:15:00.00'", cnx)

  deals = pd.DataFrame(
      [
          [np.datetime64(datetime.now()), 1000, 1000, 1000, 1, 1, 'money'],
      ],
      columns=['time', 'transaction', 'balance', 'price', 'quantity', 'is_closed', 'cause'])

  indicators = ['TrendPower', 'RSI', 'ADX', 'MACD']
  draw_figure(candles, indicators, deals)

  deals.to_csv('out.csv', index=False)
