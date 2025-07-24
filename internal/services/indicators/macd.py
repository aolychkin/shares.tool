import plotly.graph_objects as go
from ta.trend import MACD as MACD_ta


class MACD:
  """
  Класс позволяет взаимодействовать с индикатором EMA и его графиком. \n
  Для расчета нужны показатели:
  - time (index)
  - close
  """
  COLUMN_TYPES = {
      'close': 'float64',
  }

  def __init__(self, input, fast, slow, sign):
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")

    self.data = self._process_input(input)
    self._calculate(fast=fast, slow=slow, sign=sign)

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

  def _calculate(self, fast, slow, sign):
    """Вычисляет значения (обновляет self.data)."""
    ta_MACD = MACD_ta(close=self.data["close"], window_fast=fast, window_slow=slow, window_sign=sign, fillna=False)
    self.data["MACD_signal"] = ta_MACD.macd_signal()
    self.data["MACD"] = ta_MACD.macd()
    self.data["MACD_hist"] = ta_MACD.macd_diff() * 10

  # ================================================ #
  # =============== ПУБЛИЧНЫЕ МЕТОДЫ =============== #
  def get(self, columns=None):
    """
    Возвращает указанные колонки DataFrame.
    Если columns=None, возвращает:
    - 'MACD'
    - 'MACD_signal'
    - 'MACD_hist'
    """
    if columns is None:
      return self.data[['RSI', 'MACD_signal', 'MACD_hist']].copy()
    return self.data[columns].copy()

  def plot_hist(self, fig, row):
    fig.add_trace(
        go.Bar(x=self.data.index, y=self.data['MACD_hist'], name='MACD_hist', marker_color='grey'),
        row=row, col=1
    )

  def plot_line(self, fig, row, col):
    """Строит график на основе результатов."""
    fig.add_trace(
        go.Scatter(x=self.data.index, y=self.data['MACD'], mode='lines', name='MACD', marker_color='blue'),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(x=self.data.index, y=self.data['MACD_signal'], mode='lines', name='MACD_signal', marker_color='red'),
        row=row, col=col
    )
