# 자동 트레이딩 봇 (MACD + Bollinger Band + RSI + EMA 기반 + 안정성 강화)
import time
import datetime
import ccxt
import numpy as np
import json
import os
from decimal import Decimal, ROUND_DOWN

# --- 설정 ---
API_KEY = os.getenv('39f3df2c-a5b1-439a-a17f-46e01843a075')
API_SECRET = os.getenv('F1EA4B0729AD95730D8B9BCC51812054')
API_PASSWORD = os.getenv('Gyur1541@')

symbol = 'ETH-USDT-SWAP'
leverage = 10
max_risk_ratio = 0.2
target_profit_partial = 0.01
target_profit_full = 0.03
max_loss = -0.015
daily_loss_limit = -0.05
trailing_stop_trigger = 0.01
max_position_time = 60 * 60 * 6

POSITION_FILE = 'position_state.json'

# --- 거래소 연결 ---
try:
    exchange = ccxt.okx({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'password': API_PASSWORD,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    exchange.load_markets()
except Exception as e:
    print(f"[초기화 실패] 거래소 연결 오류: {e}")
    exit()

if not API_KEY or not API_SECRET or not API_PASSWORD:
    print("[주의] 환경변수 OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSWORD가 설정되지 않았습니다.")

# --- 유틸 함수 ---
def save_position(position):
    try:
        with open(POSITION_FILE, 'w') as f:
            json.dump(position, f)
    except Exception as e:
        print(f"[포지션 저장 실패] {e}")

def load_position():
    try:
        if os.path.exists(POSITION_FILE):
            with open(POSITION_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"[포지션 불러오기 실패] {e}")
    return None

def clear_position_file():
    try:
        if os.path.exists(POSITION_FILE):
            os.remove(POSITION_FILE)
    except Exception as e:
        print(f"[포지션 파일 삭제 실패] {e}")

def set_leverage(symbol):
    for _ in range(3):
        try:
            exchange.set_leverage(leverage, symbol=symbol)
            return True
        except Exception as e:
            print(f"[레버리지 설정 실패] {e}")
            time.sleep(1)
    return False

def fetch_with_retry(func, *args, retries=3, delay=1, backoff_delay=60):
    for _ in range(retries):
        try:
            return func(*args)
        except Exception as e:
            print(f"[API 호출 실패] {func.__name__}: {e}")
            if 'rate limit' in str(e).lower():
                time.sleep(backoff_delay)
            else:
                time.sleep(delay)
    return None

def adjust_position_amount(position):
    if not position:
        return None
    amount = position.get("amount", 0)
    min_amount = get_min_order_amount(symbol)
    if amount < min_amount:
        print(f"[경고] 최소 주문 수량 미만: {amount} < {min_amount}")
        return None
    return position

def get_price(symbol):
    ticker = fetch_with_retry(exchange.fetch_ticker, symbol)
    return ticker['last'] if ticker else None

def get_usdt_balance():
    balance = fetch_with_retry(exchange.fetch_balance, {'type': 'swap'})
    return balance['total'].get('USDT', 0) if balance else None

def get_min_order_amount(symbol):
    market = exchange.markets.get(symbol)
    return market['limits']['amount']['min'] if market else 0.01

def get_precision(symbol):
    market = exchange.markets.get(symbol)
    return market['precision']['amount'] if market and 'precision' in market else 4

#여기가 시간 제한 부분
def is_allowed_time():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    return not (4 <= now_kst.hour < 6)

def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    return fetch_with_retry(exchange.fetch_ohlcv, symbol, timeframe, limit)

def calc_ema(prices, period):
    if len(prices) < period:
        return np.mean(prices) if prices else 0
    prices = np.array(prices)
    k = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[period-1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        ema[i] = prices[i] * k + ema[i-1] * (1 - k)
    return ema[-1]

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_rsi(symbol):
    ohlcv = fetch_ohlcv(symbol)
    closes = [c[4] for c in ohlcv] if ohlcv else []
    return calc_rsi(closes) if len(closes) >= 15 else 50

def is_uptrend(symbol):
    ohlcv = fetch_ohlcv(symbol)
    closes = [c[4] for c in ohlcv] if ohlcv else []
    return calc_ema(closes, 5) > calc_ema(closes, 20) if len(closes) >= 20 else False

def is_downtrend(symbol):
    ohlcv = fetch_ohlcv(symbol)
    closes = [c[4] for c in ohlcv] if ohlcv else []
    return calc_ema(closes, 5) < calc_ema(closes, 20) if len(closes) >= 20 else False

def get_macd_signal(symbol):
    ohlcv = fetch_ohlcv(symbol, limit=100)
    if not ohlcv or len(ohlcv) < 35:
        return 'neutral'
    closes = [c[4] for c in ohlcv]
    short_ema = calc_ema(closes, 12)
    long_ema = calc_ema(closes, 26)
    macd_line = short_ema - long_ema
    macd_history = [calc_ema(closes[i-12:i], 12) - calc_ema(closes[i-26:i], 26) for i in range(26, len(closes))]
    if len(macd_history) < 9:
        return 'neutral'
    signal_line = calc_ema(macd_history, 9)
    if macd_line > signal_line:
        return 'buy'
    elif macd_line < signal_line:
        return 'sell'
    return 'neutral'

def get_bollinger_signal(symbol, period=20, num_std=2):
    ohlcv = fetch_ohlcv(symbol)
    if not ohlcv or len(ohlcv) < period + 1:
        return 'neutral'
    closes = [c[4] for c in ohlcv]
    ma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper = ma + num_std * std
    lower = ma - num_std * std
    current_price = closes[-1]
    if current_price > upper:
        return 'breakout_up'
    elif current_price < lower:
        return 'breakout_down'
    return 'inside_band'

def should_open_long(symbol):
    return is_uptrend(symbol) and get_rsi(symbol) > 50 and get_macd_signal(symbol) == 'buy' and get_bollinger_signal(symbol) in ['breakout_up', 'inside_band']

def should_open_short(symbol):
    return is_downtrend(symbol) and get_rsi(symbol) < 50 and get_macd_signal(symbol) == 'sell' and get_bollinger_signal(symbol) in ['breakout_down', 'inside_band']

def should_exit_long(symbol):
    return get_macd_signal(symbol) == 'sell' or get_bollinger_signal(symbol) == 'breakout_down'

def should_exit_short(symbol):
    return get_macd_signal(symbol) == 'buy' or get_bollinger_signal(symbol) == 'breakout_up'

def has_open_position(symbol):
    positions = fetch_with_retry(exchange.fetch_positions, [symbol])
    return any(float(pos.get('contracts', 0)) > 0 for pos in positions) if positions else False

def round_amount(amount, precision):
    return float(Decimal(amount).quantize(Decimal(f'1e-{precision}'), rounding=ROUND_DOWN))

def open_long(symbol, max_usdt):
    if has_open_position(symbol): return None, None
    try:
        price = get_price(symbol)
        if not price: return None, None
        precision = get_precision(symbol)
        raw_amount = (max_usdt * leverage) / price
        amount = round_amount(raw_amount, precision)
        if amount < get_min_order_amount(symbol): return None, None
        exchange.create_market_buy_order(symbol, amount)
        return price, amount
    except Exception as e:
        print(f"[롱 진입 실패] {e}")
        return None, None

def open_short(symbol, max_usdt):
    if has_open_position(symbol): return None, None
    try:
        price = get_price(symbol)
        if not price: return None, None
        precision = get_precision(symbol)
        raw_amount = (max_usdt * leverage) / price
        amount = round_amount(raw_amount, precision)
        if amount < get_min_order_amount(symbol): return None, None
        exchange.create_market_sell_order(symbol, amount)
        return price, amount
    except Exception as e:
        print(f"[숏 진입 실패] {e}")
        return None, None

def close_long(symbol, amount, reason):
    try:
        print(f"[롱 청산] {amount} ({reason})")
        exchange.create_market_sell_order(symbol, amount)
    except Exception as e:
        print(f"[롱 청산 실패] {e}")

def close_short(symbol, amount, reason):
    try:
        print(f"[숏 청산] {amount} ({reason})")
        exchange.create_market_buy_order(symbol, amount)
    except Exception as e:
        print(f"[숏 청산 실패] {e}")

def main():
    initial_balance = None
    position = load_position()
    if not set_leverage(symbol): return
    try:
        while True:
            if not is_allowed_time(): time.sleep(600); continue
            usdt_balance = get_usdt_balance()
            if usdt_balance is None: time.sleep(10); continue
            current_price = get_price(symbol)
            if current_price is None: time.sleep(10); continue
            if position is None:
                if should_open_long(symbol):
                    entry_price, amount = open_long(symbol, usdt_balance * max_risk_ratio)
                    if entry_price:
                        position = {'type': 'long', 'entry_price': entry_price, 'amount': amount, 'highest_profit': 0, 'entry_time': time.time(), 'partial_closed': False, 'entry_balance': usdt_balance}
                        save_position(position)
                elif should_open_short(symbol):
                    entry_price, amount = open_short(symbol, usdt_balance * max_risk_ratio)
                    if entry_price:
                        position = {'type': 'short', 'entry_price': entry_price, 'amount': amount, 'highest_profit': 0, 'entry_time': time.time(), 'partial_closed': False, 'entry_balance': usdt_balance}
                        save_position(position)
            else:
                pos_type = position['type']
                entry_price = position['entry_price']
                amount = position['amount']
                profit_rate = (current_price - entry_price) / entry_price if pos_type == 'long' else (entry_price - current_price) / entry_price
                if profit_rate > position['highest_profit']:
                    position['highest_profit'] = profit_rate
                total_loss = (usdt_balance - position['entry_balance']) / position['entry_balance']
                if total_loss <= daily_loss_limit or time.time() - position['entry_time'] > max_position_time or profit_rate <= max_loss or (position['highest_profit'] - profit_rate) >= trailing_stop_trigger or (pos_type == 'long' and should_exit_long(symbol)) or (pos_type == 'short' and should_exit_short(symbol)):
                    (close_long if pos_type == 'long' else close_short)(symbol, amount, "지표 기반 청산")
                    position = None
                    clear_position_file()
                elif profit_rate >= target_profit_partial and not position['partial_closed']:
                    close_amt = amount * 0.5
                    (close_long if pos_type == 'long' else close_short)(symbol, close_amt, "부분 청산")
                    position['amount'] -= close_amt
                    position['partial_closed'] = True
                    position = adjust_position_amount(position)
                    save_position(position) if position else clear_position_file()
                elif profit_rate >= target_profit_full:
                    (close_long if pos_type == 'long' else close_short)(symbol, amount, "전체 익절")
                    position = None
                    clear_position_file()
            time.sleep(2)
    except KeyboardInterrupt:
        print("[종료] 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"[치명적 오류] {e}")
        time.sleep(10)

if __name__ == '__main__':
    main()
