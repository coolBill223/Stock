from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# 下载股票数据
def download_stock_data(ticker, period="1y"):
    stock_data = yf.download(ticker, period=period)
    return stock_data

# ADX判断市场是否为震荡或趋势市场
def adx_market_type(stock_data, period=14):
    high = stock_data['High']
    low = stock_data['Low']
    close = stock_data['Close']
    
    # 计算ADX
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().clip(upper=0).abs()
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    
    # 判断市场类型
    last_adx = adx.iloc[-1]
    if last_adx < 20:
        return "震荡市场"
    elif last_adx > 25:
        return "趋势市场"
    return "无明显趋势"

# 支撑位和阻力位分析
def support_resistance_prediction(stock_data):
    recent_prices = stock_data['Close'][-30:]  # 最近30天的数据
    support_level = recent_prices.min()
    resistance_level = recent_prices.max()
    
    last_price = stock_data['Close'].iloc[-1]
    
    if last_price <= support_level * 1.02:  # 接近支撑位
        return "涨"
    elif last_price >= resistance_level * 0.98:  # 接近阻力位
        return "跌"
    return "无明确信号"

# RSI分析
def rsi_prediction(stock_data, period=14):
    delta = stock_data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    last_rsi = rsi.iloc[-1]
    
    if last_rsi < 30:  # 超卖
        return "涨"
    elif last_rsi > 70:  # 超买
        return "跌"
    return "无明确信号"

# 布林带分析
def bollinger_bands_prediction(stock_data, period=20):
    sma = stock_data['Close'].rolling(window=period).mean()
    std = stock_data['Close'].rolling(window=period).std()
    
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    
    last_price = stock_data['Close'].iloc[-1]
    
    if last_price <= lower_band.iloc[-1]:  # 接近下轨，超卖
        return "涨"
    elif last_price >= upper_band.iloc[-1]:  # 接近上轨，超买
        return "跌"
    return "无明确信号"

# 随机震荡指标分析
def stochastic_oscillator_prediction(stock_data, period=14):
    low_min = stock_data['Low'].rolling(window=period).min()
    high_max = stock_data['High'].rolling(window=period).max()
    
    slowk = 100 * (stock_data['Close'] - low_min) / (high_max - low_min)
    
    last_slowk = slowk.iloc[-1]
    
    if last_slowk < 20:  # 超卖
        return "涨"
    elif last_slowk > 80:  # 超买
        return "跌"
    return "无明确信号"

# 量价关系分析
def volume_price_relationship_prediction(stock_data):
    recent_volume = stock_data['Volume'][-1]
    avg_volume = stock_data['Volume'][-30:].mean()
    price_change = stock_data['Close'].pct_change().iloc[-1]
    
    if recent_volume > avg_volume * 1.5 and price_change > 0:  # 放量上涨
        return "涨"
    elif recent_volume > avg_volume * 1.5 and price_change < 0:  # 放量下跌
        return "跌"
    return "无明确信号"

# 简单的K线形态分析
def candlestick_pattern_prediction(stock_data):
    last_candle = stock_data['Close'].iloc[-1] - stock_data['Open'].iloc[-1]
    
    if last_candle > 0:  # 阳线
        return "涨"
    elif last_candle < 0:  # 阴线
        return "跌"
    return "无明确信号"

# 振荡区间突破分析
def breakout_prediction(stock_data):
    recent_high = stock_data['High'][-30:].max()
    recent_low = stock_data['Low'][-30:].min()
    last_price = stock_data['Close'].iloc[-1]
    
    if last_price > recent_high:  # 突破阻力位
        return "涨"
    elif last_price < recent_low:  # 跌破支撑位
        return "跌"
    return "无明确信号"

# 进行多数决策
def predict_next_day(stock_data, ticker):
    market_type = adx_market_type(stock_data)
    print(f"市场类型: {market_type}")
    
    predictions = []

    if market_type == "震荡市场":
        # 在震荡市场中，使用震荡相关的指标
        predictions.append(rsi_prediction(stock_data))
        predictions.append(bollinger_bands_prediction(stock_data))
        predictions.append(stochastic_oscillator_prediction(stock_data))
    elif market_type == "趋势市场":
        # 在趋势市场中，使用趋势相关的指标
        predictions.append(support_resistance_prediction(stock_data))
        predictions.append(volume_price_relationship_prediction(stock_data))
        predictions.append(candlestick_pattern_prediction(stock_data))
        predictions.append(breakout_prediction(stock_data))
    else:
        return "市场类型不明确，无法预测"

    # 统计涨和跌的次数
    up_count = predictions.count("涨")
    down_count = predictions.count("跌")
    
    print(f"各项预测结果: {predictions}")
    
    # 多数决策，至少2个方法预测为涨或跌
    if up_count > down_count:
        return "涨"
    elif down_count > up_count:
        return "跌"
    return "无明显趋势"

# 主函数
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prediction_class = "neutral"
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if ticker:
            try:
                stock_data = download_stock_data(ticker)
                prediction = predict_next_day(stock_data, ticker)

                if "涨" in prediction:
                    prediction_class = "up"
                elif "跌" in prediction:
                    prediction_class = "down"
                else:
                    prediction_class = "neutral"

            except Exception as e:
                prediction = f"数据获取失败: {str(e)}"
    
    return render_template('index.html', prediction=prediction, prediction_class=prediction_class)

# 启动 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)