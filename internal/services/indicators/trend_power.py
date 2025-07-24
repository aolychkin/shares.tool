# Для ADX все что не рост - то слабость тренда
import math
import pandas as pd
import plotly.graph_objects as go


class TrendPower:
  """
  Класс позволяет определить силу тренда. \n
  Для расчета нужны показатели:
  - time (index)
  - ADX (ADX, +DI, -DI)
  - EMA24
  - MACD_hist
  - RSI
  - close

  Цветовая гамма:
  - Серый = ФЛЭТ (боковик)
  - Зеленый = Сигнал на продолжение покупки
  - Оранжевый = Сигнал разворота на продажу
  - Красный = Сигнал на продолжение продажи
  - Розовый = Сигнал разворота на покупку
  """
  COLUMN_TYPES = {
      'ADX': 'float64',
      '+DI': 'float64',
      '-DI': 'float64',
      'EMA24': 'float64',
      'MACD_hist': 'float64',
      'RSI': 'float64',
      'close': 'float64',
  }

  # df[['close']].copy()
  def __init__(self, ADX, EMA, MACD_hist, RSI, close):
    input = pd.concat([ADX, EMA, MACD_hist, RSI, close], axis=1)
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")
    if total_nan_count > 0:
      input.dropna(inplace=True)

    self.data = self._process_input(input)
    self.data = self._calculate()

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

  def _calculate(self):
    """Вычисляет значения (обновляет self.data)."""
    self.data["ADX_diff"] = self.data['ADX'].diff()
    self.data["RSI_prev"] = self.data['RSI'].shift(1).fillna(0)

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
    # TODO: передать в нейронку: ADX, +DI, -DI, ADX_diff, diff(Close, EMA24), RSI_prev, RSI, RSI_diff, MACD_hist
    self.data["color"] = self.data.apply(
        lambda row:
        'grey' if (row["ADX"] < 26)
        else 'green' if (row["+DI"] > row["-DI"] and row["ADX_diff"] > 0 and row['close'] > row['EMA24'])  # and row['ADX_degrees'] >= 25 and row['ADX_degrees'] < 80
        else 'darkgreen' if (row["+DI"] > row["-DI"] and row["ADX_diff"] > 0 and row['close'] > row['EMA24'])
        else 'orange' if (row["+DI"] > row["-DI"] and row["ADX_diff"] <= 0 and row['RSI_prev'] >= 70 and row['RSI'] < 70)
        else 'yellow' if (row["+DI"] > row["-DI"] and row["ADX_diff"] <= 0)
        else 'red' if (row["+DI"] < row["-DI"] and row["ADX_diff"] > 0 and row['close'] < row['EMA24'])
        else 'darkred' if (row["+DI"] < row["-DI"] and row["ADX_diff"] > 0)
        else 'hotpink' if (row["+DI"] < row["-DI"] and row["ADX_diff"] <= 0 and row['RSI_prev'] < 30 and row['RSI'] >= 30)
        else 'pink' if (row["+DI"] < row["-DI"] and row["ADX_diff"] <= 0)
        else 'blue',
        axis=1
    )

    return self.data

  # ================================================ #
  # =============== ПУБЛИЧНЫЕ МЕТОДЫ =============== #
  def get(self, columns=None):
    """
    Возвращает указанные колонки DataFrame.
    Если columns=None, возвращает:
    - 'color'
    - 'ADX_power'
    - 'ADX_degrees'
    """
    if columns is None:
      return self.data[['color', 'ADX_power', 'ADX_degrees', 'MACD_hist']].copy()
    return self.data[columns].copy()

  def get_color(self):
    return self.data[["color"]]

  def plot_with_macd_hist(self, fig, row, col):
    """Строит график на основе результатов."""
    # if self.results is None:
    #   self.calculate()  # Автовызов, если расчет не сделан

    fig.add_trace(
        go.Bar(x=self.data.index, y=self.data['MACD_hist'], name='Trend MACD Power', marker_color=self.data["color"]),
        row=row, col=col
    )

  def plot(self, fig, row, col):
    """Строит график на основе результатов."""
    # if self.results is None:
    #   self.calculate()  # Автовызов, если расчет не сделан

    fig.add_trace(
        go.Bar(x=self.data.index, y=self.data['ADX_power'], name='Trend Power', marker_color=self.data["color"]),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=self.data.index, y=self.data['ADX_degrees'], mode='lines', name='ADX_degrees', marker_color='blue'),
        row=row, col=col
    )
