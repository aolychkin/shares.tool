from plotly.subplots import make_subplots

from internal.services.indicators import Candles, EMA, MACD, RSI, ADX, TrendPower
from internal.services.strategy import ByTrendPower


def draw_charts():
  # Отрисовка графиков индикаторов
  indicators = ['TrendPower', 'MACD_hist', 'RSI', 'ADX', 'MACD_line']

  row_heights = [0.4]
  for _ in range(len(indicators)):
    row_heights.append(0.5/len(indicators))

  fig = make_subplots(
      rows=len(indicators)+1,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.03,
      row_heights=row_heights
  )

  # Индикаторы
  candles.plot(fig=fig, row=1)
  ema55.plot(fig=fig, row=1)
  ema24.plot(fig=fig, row=1)
  ema9.plot(fig=fig, row=1)
  trend_power.plot(fig=fig, row=2)
  trend_power.plot_with_macd_hist(fig=fig, row=3)
  rsi.plot(fig=fig, row=4)
  adx.plot(fig=fig, row=5)
  macd.plot_line(fig=fig, row=6)

  # Стратегии
  deals_s1.plot_buy(fig=fig, row=1)

  # Настройка макета
  fig.update_layout(
      title_text="Цена и RSI",
      yaxis_title="Цена",
      yaxis2_title="Trend Power",
      yaxis3_title="Trend MACD Power",
      yaxis4_title="RSI",
      yaxis5_title="ADX",
      yaxis6_title="MACD",
      xaxis_rangeslider_visible=False,
      height=1300,
      # yaxis_range=[min(df['low']), max(df['high'])],
      template="plotly_dark"  # Или другой шаблон по вашему вкусу
  )
  # Отображение графика
  fig.show()


if __name__ == '__main__':
  # Формирование значений индикаторов
  candles = Candles('./storage/sqlite/shares.db')
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
      close=candles.get_column('close')
  )

  deals_s1 = ByTrendPower(
      close=candles.get_column('close'),
      trend_power=trend_power.get(),
      rsi=rsi.get(),
  )
  deals_s1.simulation()

  draw_charts()
