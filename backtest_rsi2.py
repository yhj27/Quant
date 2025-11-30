# ================================
# backtest_rsi.py (EMA Exit 제거 버전)
# ================================

import os
import time
import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import rsi
from datetime import datetime, timezone

from config import (
    SYMBOL, TIMEFRAME,
    RSI_PERIOD, BUY_RSI, SELL_RSI,
    INITIAL_CAPITAL,
    POSITION_MODE, BACKTEST_QUOTE_PER_TRADE, BACKTEST_POSITION_PCT,
    BB_WINDOW, BB_STD,
    LADDER_RSI_LEVELS,
    RSI_TP1, RSI_TP2,
    PARTIAL_SELL_RATIO,
    LADDER_USE_PRICE_FILTER, LADDER_PRICE_STEP_PCT,
    LADDER_WEIGHTS, BASE_LADDER_QUOTE,
)

# --------------------------------------
# 공통 함수
# --------------------------------------
def now_kst():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

# --------------------------------------
# 반복 호출로 과거 데이터 모두 가져오기
# --------------------------------------
def fetch_ohlcv_all(symbol, timeframe, since_iso, until_iso):
    exchange = ccxt.binance({"enableRateLimit": True})

    since = exchange.parse8601(since_iso)
    until = exchange.parse8601(until_iso)

    all_rows = []

    print(f"[{now_kst()}] {timeframe} 데이터 수집 시작...")

    while since < until:
        ohlcv = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since, limit=1000
        )

        if not ohlcv:
            break

        all_rows.extend(ohlcv)

        last_ts = ohlcv[-1][0]
        since = last_ts + 1

        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Seoul")

    print(f"[{now_kst()}] {timeframe} 데이터 수집 완료: {len(df)} rows")
    return df


# --------------------------------------
# 인디케이터: RSI + Bollinger
# --------------------------------------
def apply_15m_indicators(df):
    """
    15분봉에 RSI, Bollinger Band 추가
    EMA 기반 강제 청산은 사용하지 않으므로 ema_exit은 계산하지 않음.
    """
    df = df.copy()
    df["rsi"] = rsi(df["close"], window=RSI_PERIOD)

    ma = df["close"].rolling(BB_WINDOW).mean()
    std = df["close"].rolling(BB_WINDOW).std()
    df["bb_mid"] = ma
    df["bb_low"] = ma - BB_STD * std
    df["bb_high"] = ma + BB_STD * std

    return df.dropna()


# --------------------------------------
# 15m + 1h trend merge (1h EMA200 필터)
# --------------------------------------
def merge_trend(df_15, df_1h):
    """
    1시간봉에 EMA200을 계산하고,
    15분봉 타임스탬프에 가장 최근(과거) 1h 값을 붙인다.
    trend_ok = close_1h > ema200
    """
    df_1h = df_1h.copy()
    df_1h["ema200"] = df_1h["close"].ewm(span=200, adjust=False).mean()

    df_1h_trim = df_1h[["ts", "close", "ema200"]].rename(
        columns={"close": "close_1h"}
    )

    df_merged = pd.merge_asof(
        df_15.sort_values("ts"),
        df_1h_trim.sort_values("ts"),
        on="ts",
        direction="backward"
    )
    df_merged["trend_ok"] = df_merged["close_1h"] > df_merged["ema200"]

    return df_merged.dropna()


# --------------------------------------
# 백테스트: Trend + Bollinger + RSI + Ladder
# --------------------------------------
def backtest_rsi_portfolio_ladder(df, buy_rsi, sell_rsi):
    """
    매수 조건 (롱 only):
      - 1h EMA200 상방(trend_ok=True)  ← 장기 우상향 필터
      - 15m RSI <= LADDER_RSI_LEVELS[0]
      - 15m close <= BB_LOW
      - Ladder: LADDER_RSI_LEVELS[i] 이하 & (선택적) 가격 추가 하락 발생 시
        라더 단계별로 BASE_LADDER_QUOTE * LADDER_WEIGHTS[i] 진입

    매도 조건:
      - 1차 익절: RSI >= RSI_TP1 & 아직 partial X → PARTIAL_SELL_RATIO 만큼 청산
      - 2차 익절: RSI >= RSI_TP2 → 남은 물량 전량 청산

    ※ buy_rsi, sell_rsi 인자는 함수 시그니처 유지용 (실제 로직은 config 값 사용)
    """
    df = df.copy().reset_index(drop=True)

    cash = INITIAL_CAPITAL
    position = 0.0
    equity_list = []
    trade_rets = []
    trades = []

    entry_price = None
    entry_equity = None
    ladder_level = 0
    partial_taken = False  # 1차 익절 여부
    last_buy_price = None  # 마지막 매수가 (라더 가격 조건용)

    n_ladders = len(LADDER_RSI_LEVELS)

    for i in range(len(df) - 1):
        rsi_val     = df.loc[i, "rsi"]
        close_price = df.loc[i, "close"]
        bb_low      = df.loc[i, "bb_low"]
        trend_ok    = df.loc[i, "trend_ok"]

        next_open = df.loc[i + 1, "open"]
        ts_next   = df.loc[i + 1, "ts"]

        equity = cash + position * close_price
        equity_list.append(equity)

        # -----------------------------
        # BUY / LADDER BUY
        # -----------------------------
        buy_cond_base = (close_price <= bb_low)

        # 가격 조건: 이전 매수가 대비 LADDER_PRICE_STEP_PCT 이상 하락했는지
        if last_buy_price is None:
            price_ok_for_ladder = True  # 첫 매수는 가격 조건 없이 허용
        else:
            if LADDER_USE_PRICE_FILTER:
                price_ok_for_ladder = close_price <= last_buy_price * (1.0 - LADDER_PRICE_STEP_PCT)
            else:
                price_ok_for_ladder = True

        # 첫 매수 조건
        first_buy_cond = (
            position == 0 and trend_ok and
            rsi_val <= LADDER_RSI_LEVELS[0] and buy_cond_base
        )
        # 추가 분할매수 조건
        add_ladder_cond = (
            position > 0 and trend_ok and
            ladder_level < n_ladders and
            rsi_val <= LADDER_RSI_LEVELS[ladder_level] and
            buy_cond_base and
            price_ok_for_ladder
        )

        if first_buy_cond or add_ladder_cond:
            # === 진입 금액 결정 ===
            if POSITION_MODE == "fixed_quote":
                planned_quote = BACKTEST_QUOTE_PER_TRADE
                quote_to_use  = min(planned_quote, cash)

            elif POSITION_MODE == "fixed_pct":
                planned_quote = cash * BACKTEST_POSITION_PCT
                quote_to_use  = planned_quote

            elif POSITION_MODE == "ladder_weight":
                # 라더 단계에 따라 weight 선택
                if first_buy_cond:
                    w_idx = 0
                else:
                    w_idx = ladder_level  # ladder_level=1 → 두 번째 weight

                if w_idx >= len(LADDER_WEIGHTS):
                    w_idx = len(LADDER_WEIGHTS) - 1

                weight = LADDER_WEIGHTS[w_idx]
                planned_quote = BASE_LADDER_QUOTE * weight
                quote_to_use  = min(planned_quote, cash)
            else:
                raise ValueError("POSITION_MODE error")

            if quote_to_use > 0:
                qty = quote_to_use / next_open
                fee = 0.0005 * quote_to_use

                cash    -= (quote_to_use + fee)
                position += qty

                if first_buy_cond:
                    entry_price   = next_open
                    entry_equity  = equity
                    ladder_level  = 1
                    partial_taken = False
                else:
                    ladder_level += 1

                last_buy_price = next_open

                trades.append({
                    "side": "BUY",
                    "ts": ts_next,
                    "price": next_open,
                    "ladder_level": ladder_level,
                    "quote_used": quote_to_use,
                    "alloc_pct_of_equity": quote_to_use / equity if equity else 0,
                })

            # 매수한 캔들에서는 SELL 로직 패스
            continue

        # -----------------------------
        # SELL LOGIC (포지션 있을 때만)
        # -----------------------------
        if position > 0:
            # 1) 1차 익절: RSI >= RSI_TP1 & 아직 부분 매도 안 했을 때
            if (not partial_taken) and rsi_val >= RSI_TP1:
                sell_qty = position * PARTIAL_SELL_RATIO
                quote_received = sell_qty * next_open
                fee = 0.0005 * quote_received

                cash += (quote_received - fee)
                position -= sell_qty

                trades.append({
                    "side": "SELL_PARTIAL",
                    "ts": ts_next,
                    "price": next_open,
                    "amount": sell_qty,
                })

                partial_taken = True
                # 나머지 물량은 계속 홀딩 → 다음 캔들에서 2차 익절 체크
                continue

            # 2) 2차 익절: RSI >= RSI_TP2 → 남은 물량 전량 청산
            if rsi_val >= RSI_TP2:
                quote_received = position * next_open
                fee = 0.0005 * quote_received

                cash += (quote_received - fee)

                if entry_price is not None:
                    gross_ret = (next_open - entry_price) / entry_price
                    trade_rets.append(gross_ret)

                    trades.append({
                        "side": "SELL_FULL",
                        "ts": ts_next,
                        "price": next_open,
                        "gross_ret": gross_ret,
                        "equity_before": entry_equity,
                        "equity_after": cash,
                    })

                position = 0.0
                entry_price = None
                ladder_level = 0
                partial_taken = False
                last_buy_price = None

    # 마지막 equity
    final_price = df.loc[len(df)-1, "close"]
    equity_list.append(cash + position * final_price)
    df["equity"] = equity_list

    return df, trade_rets, trades


# --------------------------------------
# 결과 저장/출력
# --------------------------------------
def perf_summary_portfolio(trade_rets, df_equity, trades):
    if not trades:
        print("트레이드 없음")
        return

    buys = [t for t in trades if t["side"] == "BUY"]
    avg_quote = sum(t["quote_used"] for t in buys) / len(buys)

    equity = df_equity["equity"]
    final_eq = equity.iloc[-1]
    total_ret = (final_eq / INITIAL_CAPITAL) - 1
    max_dd = (equity.cummax() - equity).max() / INITIAL_CAPITAL

    rets = np.array(trade_rets)
    win_rate = (rets > 0).mean() * 100
    avg_ret = rets.mean() * 100

    print("=== BACKTEST RESULT ===")
    print(f"초기 자본: {INITIAL_CAPITAL:.2f}")
    print(f"최종 자본: {final_eq:.2f}")
    print(f"총 수익률: {total_ret*100:.2f}%")
    print(f"MDD: {max_dd*100:.2f}%")
    print(f"트레이드 횟수(완전 청산 기준): {len(trade_rets)}")
    print(f"승률: {win_rate:.2f}%")
    print(f"평균 수익률/트레이드: {avg_ret:.4f}%")
    print(f"평균 진입 금액: {avg_quote:.2f} USDT")


def save_trades_csv(trades):
    os.makedirs("logs", exist_ok=True)
    df = pd.DataFrame(trades)
    df.to_csv("logs/backtest_trades.csv", index=False, encoding="utf-8-sig")
    print("CSV 저장 완료: logs/backtest_trades.csv")


def plot_equity(df_equity):
    plt.figure(figsize=(10,6))
    plt.plot(df_equity["ts"], df_equity["equity"])
    plt.title("Equity Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------
# main
# --------------------------------------
def main():
    print(f"[{now_kst()}] 15m & 1h 데이터 다운로드 시작")

    df_15 = fetch_ohlcv_all(
        SYMBOL, TIMEFRAME,
        "2025-01-01T00:00:00Z",
        "2025-11-22T00:00:00Z"
    )
    df_1h = fetch_ohlcv_all(
        SYMBOL, "1h",
        "2024-10-01T00:00:00Z",
        "2025-11-22T00:00:00Z"
    )

    df_15 = apply_15m_indicators(df_15)
    df = merge_trend(df_15, df_1h)

    print(f"[{now_kst()}] 백테스트 시작")
    df_eq, trade_rets, trades = backtest_rsi_portfolio_ladder(df, BUY_RSI, SELL_RSI)

    perf_summary_portfolio(trade_rets, df_eq, trades)
    save_trades_csv(trades)
    plot_equity(df_eq)


if __name__ == "__main__":
    main()
