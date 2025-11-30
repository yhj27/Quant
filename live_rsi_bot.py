# live_rsi_bot.py

import os
import time
import json
import traceback
from datetime import datetime, timezone

import ccxt
import pandas as pd
from ta.momentum import rsi

from config import (
    BINANCE_API_KEY, BINANCE_SECRET,
    SYMBOL, TIMEFRAME,
    RSI_PERIOD, BUY_RSI, SELL_RSI,
    QUOTE_PER_TRADE, MAX_BASE_HOLD,
    SLEEP_SEC,
    BB_WINDOW, BB_STD,
    LADDER_RSI_LEVELS,
    TRADE_LOG_PATH, STATUS_LOG_PATH,
)

DRY_RUN = True   # 실거래 시작할 때만 False로 바꾸기

def now_kst():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")

def init_exchange():
    return ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

def fetch_ohlcv_df(exchange, symbol, timeframe, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")

    # 인디케이터: RSI + Bollinger
    df["rsi"] = rsi(df["close"], window=RSI_PERIOD)

    ma = df["close"].rolling(BB_WINDOW).mean()
    std = df["close"].rolling(BB_WINDOW).std()
    df["bb_mid"] = ma
    df["bb_low"] = ma - BB_STD * std
    df["bb_high"] = ma + BB_STD * std

    df = df.dropna()
    return df

def get_trend_filter(exchange, symbol):
    """
    1시간봉 200EMA 기준으로 추세 필터 계산
    return: (trend_ok, last_close_1h, ema200)
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=300)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")

    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    last = df.iloc[-1]
    close_1h = float(last["close"])
    ema200 = float(last["ema200"])
    trend_ok = close_1h > ema200
    return trend_ok, close_1h, ema200


def get_balances(exchange, symbol):
    base, quote = symbol.split("/")
    bal = exchange.fetch_balance()
    base_free = float(bal["free"].get(base, 0.0))
    quote_free = float(bal["free"].get(quote, 0.0))
    return base, quote, base_free, quote_free

def log_trade(data: dict):
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    df = pd.DataFrame([data])
    header = not os.path.exists(TRADE_LOG_PATH)
    df.to_csv(TRADE_LOG_PATH, mode="a", header=header, index=False, encoding="utf-8")

def log_status(msg: str):
    os.makedirs(os.path.dirname(STATUS_LOG_PATH), exist_ok=True)
    with open(STATUS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{now_kst()}] {msg}\n")

def main():
    print(f"=== RSI 25→40 + Bollinger + Ladder Live Bot 시작 (SYMBOL={SYMBOL}, TF={TIMEFRAME}, DRY_RUN={DRY_RUN}) ===")
    exchange = init_exchange()
    last_bar_time = None
    ladder_level = 0

    while True:
        try:
            df = fetch_ohlcv_df(exchange, SYMBOL, TIMEFRAME)
            if df.empty:
                log_status("캔들 데이터 없음, 대기...")
                time.sleep(SLEEP_SEC)
                continue

            trend_ok, close_1h, ema200_1h = get_trend_filter(exchange, SYMBOL)

            last = df.iloc[-1]
            prev = df.iloc[-2]

            prev_bar_time = prev["ts"]
            prev_rsi = float(prev["rsi"])
            prev_close = float(prev["close"])
            prev_bb_low = float(prev["bb_low"])

            # 새 캔들이 시작됐을 때만 의사결정
            if last_bar_time is None or prev_bar_time != last_bar_time:
                last_bar_time = prev_bar_time

                base, quote, base_free, quote_free = get_balances(exchange, SYMBOL)
                has_pos = base_free > 1e-6

                # 포지션이 없다면 ladder_level 리셋
                if not has_pos:
                    ladder_level = 0

                msg_head = (
                    f"bar={prev_bar_time}, RSI={prev_rsi:.2f}, "
                    f"close={prev_close:.2f}, bb_low={prev_bb_low:.2f}, "
                    f"trend_ok={trend_ok}, close_1h={close_1h:.2f}, ema200_1h={ema200_1h:.2f}, "
                    f"pos={has_pos}, ladder={ladder_level}, "
                    f"base_free={base_free:.6f}, quote_free={quote_free:.2f}"
                )
                print(f"[{now_kst()}] {msg_head}")
                log_status(msg_head)

                # ---------- BUY / LADDER BUY ----------
                buy_cond_base = prev_close <= prev_bb_low
                n_ladders = len(LADDER_RSI_LEVELS)

                first_buy_cond = (
                    (not has_pos) and
                    trend_ok and
                    prev_rsi <= LADDER_RSI_LEVELS[0] and
                    buy_cond_base
                )

                add_ladder_cond = (
                    has_pos and
                    trend_ok and
                    ladder_level < n_ladders and
                    prev_rsi <= LADDER_RSI_LEVELS[ladder_level] and
                    buy_cond_base
                )

                if first_buy_cond or add_ladder_cond:
                    ticker = exchange.fetch_ticker(SYMBOL)
                    last_price = float(ticker["last"])
                    quote_to_use = min(QUOTE_PER_TRADE, quote_free)

                    if quote_to_use < 10:
                        log_status(f"가용 {quote} 부족: {quote_free:.2f}, 매수 스킵")
                    else:
                        amount = quote_to_use / last_price
                        amount = float(exchange.amount_to_precision(SYMBOL, amount))
                        if amount <= 0:
                            log_status("계산된 매수 수량이 0 이하, 매수 스킵")
                        else:
                            if first_buy_cond:
                                ladder_level = 1
                            else:
                                ladder_level += 1

                            msg = (f"BUY (ladder {ladder_level}) signal: "
                                   f"RSI={prev_rsi:.2f}, close={prev_close:.2f}, "
                                   f"amount={amount}, quote={quote_to_use:.2f}")
                            print(f"[{now_kst()}] {msg}")
                            log_status(msg)

                            if not DRY_RUN:
                                order = exchange.create_order(
                                    SYMBOL, "market", "buy", amount
                                )
                                log_trade({
                                    "time": now_kst(),
                                    "side": "BUY",
                                    "symbol": SYMBOL,
                                    "amount": amount,
                                    "price": last_price,
                                    "rsi": prev_rsi,
                                    "ladder_level": ladder_level,
                                    "quote_used": quote_to_use,
                                    "raw_order": json.dumps(order),
                                })

                # ---------- SELL ----------
                elif has_pos and prev_rsi >= SELL_RSI:
                    sell_amount = min(base_free, MAX_BASE_HOLD if MAX_BASE_HOLD > 0 else base_free)
                    sell_amount = float(exchange.amount_to_precision(SYMBOL, sell_amount))

                    if sell_amount <= 0:
                        log_status("보유 수량이 없거나 너무 작아 매도 스킵")
                    else:
                        ticker = exchange.fetch_ticker(SYMBOL)
                        last_price = float(ticker["last"])
                        msg = (f"SELL signal: RSI={prev_rsi:.2f}, "
                               f"amount={sell_amount}, price≈{last_price}")
                        print(f"[{now_kst()}] {msg}")
                        log_status(msg)

                        if not DRY_RUN:
                            order = exchange.create_order(
                                SYMBOL, "market", "sell", sell_amount
                            )
                            log_trade({
                                "time": now_kst(),
                                "side": "SELL",
                                "symbol": SYMBOL,
                                "amount": sell_amount,
                                "price": last_price,
                                "rsi": prev_rsi,
                                "raw_order": json.dumps(order),
                            })
                        ladder_level = 0  # 분할매수 단계 리셋

            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            msg = f"네트워크 오류: {e}"
            print(f"[{now_kst()}] {msg}")
            log_status(msg)
            time.sleep(5)
        except ccxt.ExchangeError as e:
            msg = f"거래소 오류: {e}"
            print(f"[{now_kst()}] {msg}")
            log_status(msg)
            time.sleep(5)
        except Exception as e:
            msg = f"예외 발생: {e}"
            print(f"[{now_kst()}] {msg}")
            log_status(msg)
            traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    main()
