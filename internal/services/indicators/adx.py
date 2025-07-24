import plotly.graph_objects as go
from ta.trend import ADXIndicator


class ADX:
  """
  Класс позволяет взаимодействовать с индикатором ADX и его графиком. \nм

  Для расчета нужны показатели:
  - time (index)
  - high
  - low
  - close
  """
  COLUMN_TYPES = {
      'high': 'float64',
      'low': 'float64',
      'close': 'float64',
  }

  def __init__(self, input, period):
    total_nan_count = input.isna().sum().sum()
    print(f"Total NaN count in the {self.__class__.__name__}: {total_nan_count}")

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
    ta_ADX = ADXIndicator(high=self.data["high"], low=self.data["low"], close=self.data["close"], window=self.period, fillna=False)
    self.data["ADX"] = ta_ADX.adx()
    self.data["+DI"] = ta_ADX.adx_pos()
    self.data["-DI"] = ta_ADX.adx_neg()

  # ================================================ #
  # =============== ПУБЛИЧНЫЕ МЕТОДЫ =============== #
  def get(self, columns=None):
    """
    Возвращает указанные колонки DataFrame.
    Если columns=None, возвращает все данные.
    """
    if columns is None:
      return self.data[['ADX', '+DI', '-DI']].copy()
    return self.data[columns].copy()

  def plot(self, fig, row, col):
    """Строит график на основе результатов."""
    fig.add_trace(go.Scatter(x=self.data.index, y=self.data['ADX'], mode='lines', name='ADX', marker_color='orange'), row=row, col=col)
    fig.add_trace(go.Scatter(x=self.data.index, y=self.data['+DI'], mode='lines', name='+DI', marker_color='green'), row=row, col=col)
    fig.add_trace(go.Scatter(x=self.data.index, y=self.data['-DI'], mode='lines', name='-DI', marker_color='red'), row=row, col=col)
    fig.add_trace(go.Scatter(x=self.data.index, y=[26] * len(self.data.index), mode='lines', name='Overbought', line=dict(color='grey', dash='dash'), showlegend=False), row=row, col=col)
