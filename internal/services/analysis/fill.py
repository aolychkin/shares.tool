from internal.services.analysis import Analysis
from internal.services.strategy import ML


if __name__ == '__main__':
  # analysis = Analysis('shares_3m', 'analysis_3m', type='init')

  # # --- Инициализация всех данных по 3 и 30 минут графику --- #
  # analysis = Analysis(
  #     'shares_3m', 'analysis_3m', type='init',
  #     need_major=True, major_storage='shares_30m'
  # )
  # # analysis = Analysis('shares_3m', 'analysis_3m', type='get', all=True)

  # # Получение данных по одному дню 3 и 30 минут
  # analysis = Analysis(
  #     'shares_3m', 'analysis_3m', type='get',
  #     need_major=True,
  #     all=False, start='2025-07-10 04:00:00.000', end='2025-07-10 23:00:00.00',
  # )

  # # Получение данных по одному дню только 3 минуты
  # analysis = Analysis(
  #     'shares_3m', 'analysis_3m', type='get',
  #     all=False, start='2025-07-10 04:00:00.000', end='2025-07-10 23:00:00.00',
  # )

  #  --- Процесс обучения --- #
  version = '0.0.3'
  # # 1. Получение данных по полному набору данных 3 и 30 минут
  # analysis = Analysis(
  #     'shares_3m', 'analysis_3m', type='get',
  #     need_major=True, all=True,
  # )
  # # 2. Обучение модели и сохранение в файл
  # analysis.prepare_4_ml(version)

  # 3. Получение данных по одному дню только 3 минуты = данные для прогнозирования
  m = '07'
  d = '08'  # 11
  week = True
  if week:
    d1 = '07'
    d2 = '11'
  else:
    d1 = d
    d2 = d
  analysis = Analysis(
      'shares_3m', 'analysis_3m', type='get',
      need_major=True,
      all=False, start=f'2025-{m}-{d1} 04:00:00.000', end=f'2025-{m}-{d2} 23:00:00.00',
  )

  # 4. Прогнозирование
  analysis.predict(version)

  # 5. Отображение только 3m графика
  analysis.plot(
      main_col=[
          ['candles', 'ema55', 'ema24', 'ema9', 'deals'],
          ['tp_macd'],
          ['rsi'],
          ['adx'],
          ['macd'],
      ]
  )

  # # Отображение 3m и 30m графика
  # analysis.plot(
  #     main_col=[
  #         ['candles', 'ema55', 'ema24', 'ema9'],
  #         ['tp_macd'],
  #         ['rsi'],
  #         ['adx'],
  #         ['macd'],
  #     ],
  #     major_col=[
  #         ['candles', 'ema55', 'ema24', 'ema9'],
  #         ['tp_macd'],
  #         ['rsi'],
  #         ['adx'],
  #         ['macd'],
  #     ]
  # )

  # deals = ML(
  #     analysis.get(['ADX', '+DI', '-DI', 'MACD_hist', 'RSI', 'EMA24', 'close'])
  # )
