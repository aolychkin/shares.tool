import math
import sqlite3
import pandas as pd
import plotly.graph_objects as go


class Candles:
  """
  Класс позволяет взаимодействовать со свечами и их графиком. \n
  """
  COLUMN_TYPES = {
      'id': 'int64',
      'time': 'datetime64[ns]',
      'open': 'float64',
      'high': 'float64',
      'low': 'float64',
      'close': 'float64',
      'volume': 'int64',
      # 'is_complete': 'boolean'
  }

  def __init__(self, storage, all=False):
    cnx = sqlite3.connect(f'./storage/sqlite/{storage}.db')
    if all:
      input = pd.read_sql_query("SELECT id, time, open, high, low, close, volume FROM candles", cnx)
    else:
      input = pd.read_sql_query(
          # "SELECT id, time, open, high, low, close, volume FROM candles WHERE time >= '2025-07-10 06:00:00.000' and time <= '2025-07-10 16:00:00.00'",
          # "SELECT id, time, open, high, low, close, volume FROM candles WHERE time >= '2025-07-09 06:00:00.000' and time <= '2025-07-09 16:00:00.00'",
          # "SELECT id, time, open, high, low, close, volume FROM candles WHERE time >= '2025-07-11 06:00:00.000' and time <= '2025-07-11 16:00:00.00'",
          # "SELECT id, time, open, high, low, close, volume FROM candles WHERE time >= '2025-07-07 06:00:00.000' and time <= '2025-07-11 23:00:00.00'",
          "SELECT id, time, open, high, low, close, volume FROM candles WHERE time >= '2025-06-11 04:00:00.000' and time <= '2025-06-11 23:00:00.00'",
          cnx
      )
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")

    self.data = self._process_input(input)
    self.data.set_index('time', inplace=True)

  def _process_input(self, input):
    """Приводит данные к нужным типам."""
    # Проверка колонок
    missing = set(self.COLUMN_TYPES) - set(input.columns)
    if missing:
      raise ValueError(f"[{self.__class__.__name__}] Отсутствуют колонки: {missing}")

    # Копирование и преобразование типов
    processed = input[list(self.COLUMN_TYPES.keys())].copy()
    return processed.astype(self.COLUMN_TYPES)

  # ================================================ #
  # =============== ПУБЛИЧНЫЕ МЕТОДЫ =============== #
  def get(self, columns=None):
    if columns is None:
      return self.data[['open', 'high', 'low', 'close', 'volume']].copy()

    all_exist = all(col in self.data.columns for col in columns)
    if all_exist:
      return self.data[columns].copy()
    else:
      raise ValueError(f"[{self.__class__.__name__}] Колонки \"{columns}\" не найдены")

  def get_column(self, column):
    """Позволяет получить значение колонки по ее названию."""
    if column in self.data.columns:
      return self.data[[column]].copy()
    else:
      raise ValueError(f"[{self.__class__.__name__}] Колонка \"{column}\" не найдена")

  def get_multicolumn(self, columns):
    """Позволяет получить значение нескольких колонок по массиву названий."""
    all_exist = all(col in self.data.columns for col in columns)

    if all_exist:
      return self.data[columns].copy()
    else:
      raise ValueError(f"[{self.__class__.__name__}] Колонки \"{columns}\" не найдены")

  def plot(self, fig, row, col):
    """Строит график на основе результатов."""
    # if self.results is None:
    #   self.calculate()  # Автовызов, если расчет не сделан
    fig.add_trace(go.Candlestick(
        x=self.data.index,
        open=self.data['open'],
        high=self.data['high'],
        low=self.data['low'],
        close=self.data['close'],
        text=round((self.data["close"]/self.data["close"].shift(1)-1)*100, 2),
        name='Chandlestick'),
        row=row,
        col=col
    )
