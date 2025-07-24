import plotly.graph_objects as go
from ta.momentum import RSIIndicator


class RSI:
  """
  Класс позволяет взаимодействовать с индикатором RSI и его графиком. \n
  - Сигналом является только выход из зоны 70/30, вход и нахождение в зоне сигналом НЕ является
  - Нам интересна только та свеча, на которой возник сигнал 70/30. Если ADX положительный и растущий, то ничего не делаем при этом

  Для расчета нужны показатели:
  - time (index)
  - close
  """
  COLUMN_TYPES = {
      'close': 'float64',
  }

  # TODO: Добавить учет угла RSI (или дельту от 70, чтобы избежать мильтешения)
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
    self.data['RSI'] = RSIIndicator(close=self.data['close'], window=self.period, fillna=False).rsi()

  # ================================================ #
  # =============== ПУБЛИЧНЫЕ МЕТОДЫ =============== #
  def get(self, columns=None):
    """
    Возвращает указанные колонки DataFrame.
    Если columns=None, возвращает:
    - 'RSI'
    """
    if columns is None:
      return self.data[['RSI']].copy()
    return self.data[columns].copy()

  def plot(self, fig, row):
    """Строит график на основе результатов."""
    fig.add_trace(
        go.Scatter(x=self.data.index, y=self.data['RSI'], mode='lines', name=f'{self.__class__.__name__}', marker_color='orange'),
        row=row, col=1
    )
    # Добавление уровней перекупленности/перепроданности
    fig.add_trace(
        go.Scatter(
            x=self.data.index,
            y=[70] * len(self.data.index),
            mode='lines',
            name='Overbought',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=row, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=self.data.index,
            y=[30] * len(self.data.index),
            mode='lines',
            name='Oversold',
            line=dict(color='green', dash='dash'),
            showlegend=False),
        row=row, col=1
    )
