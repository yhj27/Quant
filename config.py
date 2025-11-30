# config.py
import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")

if not BINANCE_API_KEY or not BINANCE_SECRET:
    raise RuntimeError("환경 변수에 BINANCE_API_KEY / BINANCE_SECRET를 설정하세요 (.env 파일).")

# 기본 전략/시장 설정
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"   # 5m, 15m, 1h 등

RSI_PERIOD = 14
BUY_RSI = 25
SELL_RSI = 45

# --- 포트폴리오 백테스트 설정 ---
INITIAL_CAPITAL = 100.0          # 백테스트용 초기 자본

# --- 라이브 봇 설정 ---
QUOTE_PER_TRADE = 50.0            # 실시간 봇 1회 매수 USDT
MAX_BASE_HOLD = 0.01              # spot에서 최대 보유 코인 수 (안전장치)

# --- 볼린저 밴드 설정 ---
BB_WINDOW = 20                    # 볼린저 계산용 기간
BB_STD = 2.0                      # 표준편차 배수 (보통 2)

# --- 분할매수(RSI 레벨) ---
# 첫 매수: 25 이하, 2단계: 20 이하, 3단계: 15 이하에서 추가매수
LADDER_RSI_LEVELS = [25, 25, 25]  # 필요하면 자유롭게 조정

# --- 라더 매수 크기 weight ---
#===================================================
# 1차: 1.0배, 2차: 1.5배, 3차: 2.0배
LADDER_WEIGHTS = [1.0, 2.0, 1.5]

# 라더 1단계 기준 금액 (기본 단위)
BASE_LADDER_QUOTE = 20.0   # 1차 매수 50$, 2차 75$, 3차 100$가 됨
LADDER_USE_PRICE_FILTER = True
LADDER_PRICE_STEP_PCT = 0.001   # 1%씩 더 싸졌을 때만 다음 라더 (예: 0.01 = 1%)


POSITION_MODE = "ladder_weight"   # "fixed_quote", "fixed_pct" 대신 이걸 사용할 예정
BACKTEST_QUOTE_PER_TRADE = 50.0   # BASE_LADDER_QUOTE와 같은 값으로 두거나 무시해도 됨
BACKTEST_POSITION_PCT = 0.3       # 이제는 안 쓰이지만 그대로 둬도 문제 없음
#===================================================

# RSI 기반 부분 익절/전체 익절
RSI_TP1 = 70          # 1차 익절 RSI (부분 청산)
RSI_TP2 = 70          # 2차 익절 RSI (전체 청산)

# 1차 익절 때 몇 % 비중을 파는지
PARTIAL_SELL_RATIO = 0.7   # 0.5 = 50% 청산

# EMA 강제 청산용 (15분봉 기준)
EMA_EXIT_PERIOD = 30       # 15m EMA 기간
USE_EMA_EXIT = True        # True면 close < EMA_EXIT 시 전량 청산
