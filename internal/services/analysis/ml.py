from datetime import datetime, time
import math
import re
import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from internal.services.indicators import Candles, EMA, MACD, RSI, ADX, TrendPower


class AnalXXysis:
  """
  Собирает все возможные показатели в единую таблицу
  """
  COLUMN_TYPES = {
      # Candles
      'open': 'float64',
      'high': 'float64',
      'low': 'float64',
      'close': 'float64',
      'volume': 'int64',
      # ADX
      'ADX': 'float64',
      '+DI': 'float64',
      '-DI': 'float64',
      # EMA
      'EMA55': 'float64',
      'EMA24': 'float64',
      'EMA9': 'float64',
      # MACD
      'MACD': 'float64',
      'MACD_signal': 'float64',
      'MACD_hist': 'float64',
      # RSI
      'RSI': 'float64',
      # Trend power
      'color': 'string',
      'ADX_degrees': 'float64',
  }
  COLUMN_TYPES_MAJOR = {
      # Candles
      'open': 'float64',
      'high': 'float64',
      'low': 'float64',
      'close': 'float64',
      'volume': 'int64',
      'open_major': 'float64',
      'high_major': 'float64',
      'low_major': 'float64',
      'close_major': 'float64',
      'volume_major': 'int64',
      # ADX
      'ADX': 'float64',
      '+DI': 'float64',
      '-DI': 'float64',
      'ADX_major': 'float64',
      '+DI_major': 'float64',
      '-DI_major': 'float64',
      # EMA
      'EMA55': 'float64',
      'EMA24': 'float64',
      'EMA9': 'float64',
      'EMA55_major': 'float64',
      'EMA24_major': 'float64',
      'EMA9_major': 'float64',
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
      'color': 'string',
      'ADX_degrees': 'float64',
      'color_major': 'string',
      'ADX_degrees_major': 'float64',
  }

  def __init__(
      self, candle_storage, analysis_storage, type='get',
      # в любых случаях, если необходимо подгрузить больший таймфрейм
      need_major=False,
      # только если type='init'
      major_storage='',
      # только если type='get'
      all=False,  # все даты или только определенные?
      start='2025-07-10 04:00:00.000', end='2025-07-10 23:00:00.00',
  ):
    self.major = need_major
    if type == 'init':
      if self.major:
        input_minor = self._calculate_indicators_from_db(candle_storage=candle_storage)
        input_major = self._calculate_major_indicators_from_db(major_storage=major_storage)
        input = pd.merge_asof(
            left=input_minor.reset_index(),  # 3-минутные данные
            right=input_major.reset_index(),  # 30-минутные данные
            on='time',  # Ключ для сопоставления
            direction='backward'  # Берем последнюю подходящую 30-минутную запись
        ).set_index('time')
        print(input.head)
      else:
        input = self._calculate_indicators_from_db(candle_storage=candle_storage)
    else:
      cnx = sqlite3.connect(f'./storage/sqlite/{analysis_storage}.db')
      if all:
        input = pd.read_sql_query(f'SELECT * FROM {analysis_storage}', cnx)
      else:
        input = pd.read_sql_query(
            f"SELECT * FROM {analysis_storage} WHERE time >= '{start}' and time <= '{end}'",
            cnx
        )
      input['time'] = pd.to_datetime(input['time'])
      input.set_index('time', inplace=True)
      print(input.head)

    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")
    if total_nan_count > 0:
      input.dropna(inplace=True)
    self.data = self._process_input(input)
    self._calculate_candlestick('')
    if self.major:
      self._calculate_candlestick('_major')

    # round((draw_data[f'close{suffix}']/draw_data[f'close{suffix}'].shift(1)-1)*100, 2)

    if type == 'init':
      print(self.data.head)
      self._save_indicators_to_db(analysis_storage)

  # def _delta(self, old, new):
  #   return ((new - old)/old*100)

  # def _average(self, row, suffix):
  #   return np.mean([row[f'open{suffix}'], row[f'close{suffix}']])

  def _calculate_candlestick(self, suffix):
    # TODO: возможно просто заменить на change (a_open. a_close)
    self.data[f'change{suffix}'] = self._delta(
        self.data[f'open{suffix}'],
        self.data[f'close{suffix}']
    )
    self.data[f'a_low{suffix}'] = self._delta(
        (self.data[f'open{suffix}'] + self.data[f'close{suffix}'])/2,
        self.data[f'low{suffix}']
    ).abs()
    self.data[f'a_high{suffix}'] = self._delta(
        (self.data[f'open{suffix}'] + self.data[f'close{suffix}'])/2,
        self.data[f'high{suffix}']
    ).abs()
    self.data[f'tail_low{suffix}'] = self._delta(
        self.data[[f'open{suffix}', f'close{suffix}']].min(axis=1),
        self.data[f'low{suffix}']
    ).abs()
    self.data[f'tail_high{suffix}'] = self._delta(
        self.data[[f'open{suffix}', f'close{suffix}']].max(axis=1),
        self.data[f'high{suffix}']
    ).abs()

  def _delta_col(self, old, new):
    return ((new - old)/old*100)

  def _process_input(self, input):
    """Приводит данные к нужным типам."""
    # Проверка колонок
    if self.major:
      missing = set(self.COLUMN_TYPES_MAJOR) - set(input.columns)
    else:
      missing = set(self.COLUMN_TYPES) - set(input.columns)

    if missing:
      raise ValueError(f"[{self.__class__.__name__}] Отсутствуют колонки: {missing}")
    if str(input.index.dtype) != 'datetime64[ns]':
      raise ValueError(f"[{self.__class__.__name__}] Отсутствуют шкала времени, в качестве индекса: time")

    # Копирование и преобразование типов
    if self.major:
      processed = input[list(self.COLUMN_TYPES_MAJOR.keys())].copy()
      return processed.astype(self.COLUMN_TYPES_MAJOR)
    else:
      processed = input[list(self.COLUMN_TYPES.keys())].copy()
      return processed.astype(self.COLUMN_TYPES)

  def _save_indicators_to_db(self, analysis_storage):
    engine = create_engine(f'sqlite:///storage/sqlite/{analysis_storage}.db')
    engine.connect()
    self.data.to_sql(
        name=f'{analysis_storage}',          # Название таблицы
        con=engine,                                   # Подключение SQLAlchemy
        if_exists='replace',                          # 'replace' (перезапись), 'append' (добавление), 'fail' (ошибка)
        index=True                                    # True = сохранять индекс DataFrame
    )

  def _calculate_major_indicators_from_db(self, major_storage):
    candles = Candles(storage=major_storage, all=True)
    ema55 = EMA(candles.get_column('close'), 55)
    ema24 = EMA(candles.get_column('close'), 24)
    ema9 = EMA(candles.get_column('close'), 9)
    macd = MACD(input=candles.get_column('close'), fast=12, slow=26, sign=9)
    rsi = RSI(candles.get_column('close'), 14)
    adx = ADX(candles.get_multicolumn(['high', 'low', 'close']), 14)
    trend_power = TrendPower(
        ADX=adx.get(),
        EMA=ema24.get(),
        MACD_hist=macd.get('MACD_hist'),
        RSI=rsi.get(),
        close=candles.get_column('close')
    )
    input = pd.concat(
        [
            candles.get(),
            ema55.get(),
            ema24.get(),
            ema9.get(),
            macd.get(),
            rsi.get(),
            adx.get(),
            trend_power.get(['color', 'ADX_degrees']),
        ], axis=1)
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")
    if total_nan_count > 0:
      input.dropna(inplace=True)
    return input.add_suffix('_major')

  def _calculate_indicators_from_db(self, candle_storage):
    candles = Candles(storage=candle_storage, all=True)
    ema55 = EMA(candles.get_column('close'), 55)
    ema24 = EMA(candles.get_column('close'), 24)
    ema9 = EMA(candles.get_column('close'), 9)
    macd = MACD(input=candles.get_column('close'), fast=12, slow=24, sign=10)
    rsi = RSI(candles.get_column('close'), 9)
    adx = ADX(candles.get_multicolumn(['high', 'low', 'close']), 9)
    trend_power = TrendPower(
        ADX=adx.get(),
        EMA=ema24.get(),
        MACD_hist=macd.get('MACD_hist'),
        RSI=rsi.get(),
        close=candles.get_column('close')
    )
    input = pd.concat(
        [
            candles.get(),
            ema55.get(),
            ema24.get(),
            ema9.get(),
            macd.get(),
            rsi.get(),
            adx.get(),
            trend_power.get(['color', 'ADX_degrees']),
        ], axis=1)
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")
    if total_nan_count > 0:
      input.dropna(inplace=True)
    return input

  def _delta(self, old, new):
    return ((new - old)/old*100)

  def _average(self, row, suffix):
    return np.mean([row[f'open{suffix}'], row[f'close{suffix}']])

  def _draw_by_col(self, rows, col, is_major=False):
    suffix = ''
    draw_data = self.data.copy()
    if is_major:
      suffix = '_major'
      draw_data = self.data[self.data.index.minute % 30 == 0].copy()

    draw_data[f'close{suffix}_prev'] = draw_data[f'close{suffix}'].shift(1)

    draw_data[f'tooltip{suffix}'] = draw_data.apply(
        lambda row: f"<br>"
        f"RSI: {row[f'RSI{suffix}']:.4f}<br>"
        f"MACD: {row[f'MACD{suffix}']:.4f}<br>"
        f"Average: {self._average(row, suffix):.4f}<br>"
        f"a_low:  {row[f'a_low{suffix}']:.4f}%<br>"
        f"a_high:  {row[f'a_high{suffix}']:.4f}%<br>"
        f"tail_low:  {row[f'tail_low{suffix}']:.4f}%<br>"
        f"tail_high:  {row[f'tail_high{suffix}']:.4f}%<br>"
        f"Change: {row[f'change{suffix}']:.4f}%",
        axis=1
    )

    for row, line in enumerate(rows):
      for plot in line:
        match plot:
          case 'adx':
            self.fig.add_trace(go.Scatter(
                x=draw_data.index,
                y=draw_data[f'ADX{suffix}'],
                mode='lines', name='ADX', marker_color='orange'),
                row=row+1, col=col
            )
            self.fig.add_trace(go.Scatter(
                x=draw_data.index,
                y=draw_data[f'+DI{suffix}'],
                mode='lines', name='+DI', marker_color='green'),
                row=row+1, col=col
            )
            self.fig.add_trace(go.Scatter(
                x=draw_data.index,
                y=draw_data[f'-DI{suffix}'],
                mode='lines', name='-DI', marker_color='red'),
                row=row+1, col=col
            )
            self.fig.add_trace(go.Scatter(
                x=draw_data.index,
                y=[26] * len(draw_data.index),
                mode='lines', name='Overbought', line=dict(color='grey', dash='dash'), showlegend=False),
                row=row+1, col=col
            )
          case 'candles':
            self.fig.add_trace(go.Candlestick(
                x=draw_data.index,
                open=draw_data[f'open{suffix}'],
                high=draw_data[f'high{suffix}'],
                low=draw_data[f'low{suffix}'],
                close=draw_data[f'close{suffix}'],
                text=draw_data[f'tooltip{suffix}'],
                name='Chandlestick'),
                row=row+1,
                col=col
            )
          case _ if re.match(r'^ema(\d+)$', plot):
            period = int(re.match(r'^ema(\d+)$', plot).group(1))
            if period < 15:
              color = 'red'
            elif period < 40:
              color = 'blue'
            else:
              color = 'yellow'
            self.fig.add_trace(
                go.Scatter(
                    x=draw_data.index,
                    y=draw_data[f'EMA{period}{suffix}'],
                    mode='lines',
                    name=f'EMA{period}',
                    marker_color=color),
                row=row+1,
                col=col
            )
          case 'macd':
            self.fig.add_trace(
                go.Scatter(
                    x=draw_data.index,
                    y=draw_data[f'MACD{suffix}'],
                    mode='lines', name='MACD', marker_color='blue'
                ),
                row=row+1, col=col
            )
            self.fig.add_trace(
                go.Scatter(
                    x=draw_data.index,
                    y=draw_data[f'MACD_signal{suffix}'],
                    mode='lines', name='MACD_signal', marker_color='red'
                ),
                row=row+1, col=col
            )
          case 'macd_hist':
            self.fig.add_trace(
                go.Bar(
                    x=draw_data.index,
                    y=draw_data[f'MACD_hist{suffix}'],
                    name='MACD_hist', marker_color='grey'
                ),
                row=row+1, col=col
            )
          case 'rsi':
            self.fig.add_trace(
                go.Scatter(
                    x=draw_data.index,
                    y=draw_data[f'RSI{suffix}'],
                    mode='lines', name='RSI', marker_color='orange'
                ),
                row=row+1, col=col
            )
            # Добавление уровней перекупленности/перепроданности
            self.fig.add_trace(
                go.Scatter(
                    x=draw_data.index,
                    y=[70] * len(draw_data.index),
                    mode='lines',
                    name='Overbought',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=row+1, col=col
            )
            self.fig.add_trace(
                go.Scatter(
                    x=draw_data.index,
                    y=[30] * len(draw_data.index),
                    mode='lines',
                    name='Oversold',
                    line=dict(color='green', dash='dash'),
                    showlegend=False),
                row=row+1, col=col
            )
          case 'tp_macd':
            self.fig.add_trace(
                go.Bar(
                    x=draw_data.index,
                    y=draw_data[f'MACD_hist{suffix}'],
                    name='Trend MACD Power', marker_color=draw_data["color"]
                ),
                row=row+1, col=col
            )
            self.fig.add_trace(
                go.Scatter(
                    x=draw_data.index,
                    y=draw_data[f'ADX_degrees{suffix}'],
                    mode='lines', name='ADX_degrees', marker_color='blue'
                ),
                row=row+1, col=col
            )

  # ================================================ #
  # =============== ПУБЛИЧНЫЕ МЕТОДЫ =============== #
  def get(self, columns=None):
    if columns is None:
      return self.data.copy()

    all_exist = all(col in self.data.columns for col in columns)
    if all_exist:
      return self.data[columns].copy()
    else:
      raise ValueError(f"[{self.__class__.__name__}] Колонки \"{columns}\" не найдены")

  def prepare_4_ml(self):
    features = [
        # ADX
        'ADX',
        '+DI',
        '-DI',
        'ADX_major',
        '+DI_major',
        '-DI_major',
        # EMA
        'EMA55',
        'EMA24',
        'EMA9',
        'EMA55_major',
        'EMA24_major',
        'EMA9_major',
        # MACD
        'MACD',
        'MACD_signal',
        'MACD_hist',
        'MACD_major',
        'MACD_signal_major',
        'MACD_hist_major',
        # RSI
        'RSI',
        'RSI_major',
        # Trend power
        'ADX_degrees',
        'ADX_degrees_major',
    ]
    pass

  def plot(self, main_col=None, major_col=None):
    row_heights = [0.4]
    if main_col != None and major_col != None:
      max_rows = max(len(main_col), len(major_col))
      for _ in range(max_rows-1):
        row_heights.append((1-row_heights[0])/max_rows)
      self.fig = make_subplots(
          rows=max_rows,
          cols=2,
          shared_xaxes=True,
          vertical_spacing=0.03,
          row_heights=row_heights
      )
      self._draw_by_col(rows=main_col, col=1)
      # self._draw_by_col(rows=main_col, col=2)
      self._draw_by_col(rows=major_col, col=2, is_major=True)
    else:
      for _ in range(len(main_col)):
        row_heights.append((1-row_heights[0])/len(main_col))
      self.fig = make_subplots(
          rows=len(main_col)+1,
          cols=1,
          shared_xaxes=True,
          vertical_spacing=0.03,
          row_heights=row_heights
      )
      self._draw_by_col(rows=main_col, col=1)

    self.fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=1300,
        # yaxis_range=[min(df['low']), max(df['high'])],
        template="plotly_dark",
        xaxis=dict(
            rangebreaks=[
                dict(bounds=[21, 4], pattern='hour'),  # Пропускать ночь (21:00-9:00)
                # dict(bounds=['sat', 'mon'])            # Пропускать выходные
            ]
        )
    )
    # Отображение графика
    self.fig.show()
