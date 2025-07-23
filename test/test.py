import pandas as pd
import numpy as np
import ta  # Для технических индикаторов
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf  # Для загрузки данных (можно заменить на свои данные)

# Загрузка данных (пример: акции S&P 500)
data = yf.download('SPY', start='2023-01-01', end='2024-01-01')
data = data.reset_index()

# Расчет индикаторов
# 1. EMA (скользящая средняя)
data['EMA'] = ta.trend.ema_indicator(data['Close'], window=20)

# 2. ADX (Average Directional Movement Index)
data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=14)

# 3. +DI и -DI (Directional Indicators)
data['+DI'] = ta.trend.plus_di(data['High'], data['Low'], data['Close'], window=14)
data['-DI'] = ta.trend.minus_di(data['High'], data['Low'], data['Close'], window=14)

# 4. RSI (Relative Strength Index)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# Генерация сигналов
# Сигналы продолжения тренда (покупка/продажа)
data['Buy_Signal'] = (data['Close'] > data['EMA']) & (data['ADX'] > 25) & (data['+DI'] > data['-DI'])
data['Sell_Signal'] = (data['Close'] < data['EMA']) & (data['ADX'] > 25) & (data['-DI'] > data['+DI'])

# Сигналы разворота (закрытие позиций)
data['Exit_Buy_Signal'] = (data['RSI'] < 30) & (data['ADX'].diff() < 0)  # RSI < 30 + ADX снижается
data['Exit_Sell_Signal'] = (data['RSI'] > 70) & (data['ADX'].diff() < 0)  # RSI > 70 + ADX снижается

# Визуализация с помощью Plotly
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=('Цена и EMA', 'ADX и DI', 'RSI'),
                    vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])

# 1. График цены и EMA
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Цена', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA'], name='EMA (20)', line=dict(color='orange')), row=1, col=1)

# Пометки сигналов покупки (зеленые стрелки вверх)
fig.add_trace(go.Scatter(
    x=data[data['Buy_Signal']]['Date'],
    y=data[data['Buy_Signal']]['Close'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=10, color='green'),
    name='Покупка (Trend)'
), row=1, col=1)

# Пометки сигналов продажи (красные стрелки вниз)
fig.add_trace(go.Scatter(
    x=data[data['Sell_Signal']]['Date'],
    y=data[data['Sell_Signal']]['Close'],
    mode='markers',
    marker=dict(symbol='triangle-down', size=10, color='red'),
    name='Продажа (Trend)'
), row=1, col=1)

# 2. График ADX и DI
fig.add_trace(go.Scatter(x=data['Date'], y=data['ADX'], name='ADX', line=dict(color='purple')), row=2, col=1)
fig.add_trace(go.Scatter(x=data['Date'], y=data['+DI'], name='+DI', line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=data['Date'], y=data['-DI'], name='-DI', line=dict(color='red')), row=2, col=1)

# 3. График RSI
fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', line=dict(color='blue')), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)  # Уровень перекупленности
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)  # Уровень перепроданности

# Разметка
fig.update_layout(
    title='Торговая стратегия: EMA + ADX + RSI',
    hovermode='x unified',
    height=800,
    showlegend=True
)

fig.show()
