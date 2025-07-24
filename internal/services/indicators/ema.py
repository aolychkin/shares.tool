import plotly.graph_objects as go
from ta.trend import EMAIndicator


class EMA:
  """
  Класс позволяет взаимодействовать с индикатором EMA и его графиком. \n
  Для расчета нужны показатели:
  - time (index)
  - close
  """
  COLUMN_TYPES = {
      'close': 'float64',
  }

  def __init__(self, input, period):
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}{period}: {total_nan_count}")

    self.data = self._process_input(input)
    self.period = period
    self._calculate()

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
    self.data[f'EMA{self.period}'] = EMAIndicator(close=self.data["close"], window=self.period, fillna=False).ema_indicator()

  # ================================================ #
  # =============== ПУБЛИЧНЫЕ МЕТОДЫ =============== #
  def get(self, columns=None):
    """
    Возвращает указанные колонки DataFrame.
    Если columns=None, возвращает все данные.
    """
    if columns is None:
      return self.data[[f'EMA{self.period}']].copy()
    return self.data[columns].copy()

  def plot(self, fig, row):
    """Строит график на основе результатов."""
    if self.period < 15:
      color = 'red'
    elif self.period < 40:
      color = 'blue'
    else:
      color = 'yellow'

    fig.add_trace(
        go.Scatter(
            x=self.data.index,
            y=self.data[f'EMA{self.period}'],
            mode='lines',
            name=f'EMA{self.period}',
            marker_color=color),
        row=row,
        col=1
    )
