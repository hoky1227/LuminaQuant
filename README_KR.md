[English Version](README.md)

# LuminaQuant (루미나 퀀트) - 이벤트 기반 퀀트 트레이딩 시스템

Binance에서의 안정적인 백테스팅과 라이브 트레이딩을 위해 설계된 고급 이벤트 기반 퀀트 트레이딩 파이프라인입니다. 시뮬레이션과 실제 실행 간의 일관성을 보장하기 위해 모듈식 아키텍처로 구축되었습니다.

## 주요 기능

- **이벤트 기반 아키텍처**: 실제 트레이딩 메커니즘을 모방하기 위해 중앙 집중식 이벤트 루프(`MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`)를 사용합니다.
- **정확한 백테스팅**:
  - **Trailing Stops**, Stop Loss, Take Profit 지원.
  - 체결 지연(latency) 및 수수료 시뮬레이션.
- **라이브 트레이딩**:
  - `ccxt`를 통한 **Binance** 연동.
  - **강력한 상태 관리**: 시작 시 거래소와 포지션 동기화.
  - **부분 체결**: 미체결 수량을 큐에 넣어 유동성 제약을 처리.
- **최적화 (Optimization)**:
  - **Optuna**(베이지안 최적화) 또는 Grid Search를 사용한 파라미터 튜닝 내장.
  - 전진 분석 (Walk-Forward Analysis) 지원 (Train/Validation/Test 분할).
- **대화형 대시보드 (Interactive Dashboard)**:
  - Streamlit 기반의 대시보드로 성과 분석, 수익 곡선, 매매 시점 시각화 제공.
- **간편한 설정**:
  - `config.yaml` 중앙 설정 파일 하나로 모든 트레이딩, 백테스트, 최적화 제어 가능.

## 프로젝트 구조

```
lumina-quant/
├── lumina_quant/           # 코어 패키지
│   ├── backtest.py         # 백테스팅 엔진
│   ├── live_trader.py      # 라이브 트레이딩 엔진
│   └── config.py           # 설정 로더
├── generate_data.py        # 합성 데이터 생성기
├── run_backtest.py         # 백테스트 실행 스크립트
├── optimize.py             # 전략 최적화 (Optuna)
├── run_live.py             # 라이브 트레이딩 실행 스크립트
├── dashboard.py            # Streamlit 대시보드
├── config.yaml             # 메인 설정 파일
└── README_KR.md
```

## 설치 및 설정

1.  **의존성 설치**
    ```bash
    uv sync
    # 또는
    pip install .
    ```

2.  **설정 (Configuration)**
    
    **메인 설정**: `config.yaml`을 수정하여 거래 심볼, 타임프레임, 리스크 파라미터 등을 설정합니다.
    
    **비밀 키 (Secrets)**: `.env.example`을 `.env`로 복사하고 API 키를 설정합니다 (라이브 트레이딩 시 필수).
    ```bash
    cp .env.example .env
    ```
    `.env` 편집:
    ```ini
    BINANCE_API_KEY=your_key
    BINANCE_SECRET_KEY=your_secret
    ```

## 설정 상세 가이드 (`config.yaml`)

`config.yaml` 파일이 시스템의 핵심 제어 센터입니다. 각 항목별 상세 설명입니다:

### `system` (시스템)
- `log_level`: 로그 출력 레벨 (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

### `trading` (공통 트레이딩 설정)
- `symbols`: 거래할 코인 쌍 리스트 예: `["BTC/USDT", "ETH/USDT"]`.
- `timeframe`: 캔들 주기. 예: `1m`, `5m`, `1h`, `4h`, `1d`.
- `initial_capital`: 시뮬레이션 시작 자본금.
- `target_allocation`: 포지션 사이즈 규칙. `0.1`은 전체 자산의 10%를 한 종목에 진입함을 의미.
- `min_trade_qty`: 최소 거래 수량 (거래소 오류 방지).

### `backtest` (백테스트 설정)
- `start_date`: 시뮬레이션 시작 날짜 (`YYYY-MM-DD`).
- `end_date`: 시뮬레이션 종료 날짜 (`null`이면 데이터 끝까지 실행).
- `commission_rate`: 편도 거래 수수료. `0.001` = 0.1% (바이낸스 기본값).
- `slippage_rate`: 슬리피지 모델링. `0.0005` = 0.05% (체결 오차).
- `annual_periods`: 연환산 지표 계산용 주기.
    - 코인 24시간 일일 데이터: `365`
    - 코인 시간봉: `8760` (365 * 24)
    - 코인 분봉: `525600` (365 * 24 * 60)

### `live` (라이브 트레이딩)
- `testnet`: `true`면 바이낸스 테네넷(모의투자), `false`면 실계좌 사용.
- `poll_interval`: 데이터 조회 및 루프 실행 간격 (초 단위).
- `order_timeout`: 주문 미체결 시 취소/재시도 대기 시간.

### `optimization` (최적화)
- `method`: `OPTUNA` (추천) 또는 `GRID`.
- `strategy`: 최적화할 전략 클래스명 (예: `RsiStrategy`).
- `optuna`:
    - `n_trials`: 시도할 횟수.
    - `params`: 튜닝할 파라미터 범위 정의.
        - 타입: `int`, `float`, `categorical`.
        - 키 이름은 전략 클래스 `__init__`의 인자와 일치해야 함.

## 대화형 대시보드 (Interactive Dashboard)

LuminaQuant에는 백테스트 결과를 시각화할 수 있는 대시보드가 내장되어 있습니다.

### 대시보드 실행
백테스트(`run_backtest.py`) 실행 후, 다음 명령어를 입력하세요:
```bash
uv run streamlit run dashboard.py
```

### 주요 기능
1.  **성과 지표**: 총 수익률(ROI), 최대 낙폭(MDD), 최종 자산, 총 거래 횟수를 실시간으로 표시.
2.  **가격 및 매매 차트**: 자산 가격 그래프 위에 매수(초록색 삼각형), 매도(빨간색 삼각형) 시점을 표시.
3.  **수익 곡선 (Equity Curve)**: 내 전략의 자산 성장 그래프 vs 벤치마크(Buy & Hold) 비교.
4.  **낙폭 분석 (Drawdown Analysis)**: "Underwater" 차트를 통해 자산 하락의 깊이와 기간을 시각화.
5.  **최적 파라미터**: 최적화 도구를 통해 찾은 베스트 파라미터를 표시.

## 전략 개발 가이드 (Strategy Development)

### 새 전략 만들기

1.  `strategies/` 폴더에 새 파일을 만듭니다 (예: `custom_strategy.py`).
2.  `lumina_quant.strategy.Strategy`를 상속받습니다.
3.  `__init__`에서 `bars`, `events` 및 파라미터를 받도록 구현합니다.
4.  `calculate_signals(self, event)` 메서드에서 매수/매도 로직을 구현합니다.

**예제 템플릿:**

```python
from lumina_quant.strategy import Strategy
from lumina_quant.events import SignalEvent

class CustomStrategy(Strategy):
    def __init__(self, bars, events, my_param=10):
        self.bars = bars      # 데이터 핸들러
        self.events = events  # 이벤트 큐
        self.my_param = my_param
        self.symbol_list = self.bars.symbol_list

    def calculate_signals(self, event):
        if event.type == "MARKET":
            for s in self.symbol_list:
                # 최근 종가 가져오기
                bars = self.bars.get_latest_bars_values(s, "close", N=self.my_param)
                if len(bars) < self.my_param:
                    continue
                
                # 로직 구현
                if bars[-1] > bars[-2]:
                    # SignalType: "LONG", "SHORT", "EXIT"
                    signal = SignalEvent(1, s, event.time, "LONG", 1.0)
                    self.events.put(signal)
```

### 전략 등록하기

1.  `run_backtest.py`와 `optimize.py`에서 작성한 전략을 import합니다.
2.  `config.yaml`에서 사용할 수 있도록 `STRATEGY_MAP`에 추가합니다.

## 파라미터 튜닝 (Optimization)

`config.yaml`을 통해 전략 파라미터를 최적화할 수 있습니다.

**`CustomStrategy` 최적화 설정 예시:**

```yaml
optimization:
  method: "OPTUNA"
  strategy: "CustomStrategy"
  
  optuna:
    n_trials: 50
    params:
      my_param:
        type: "int"
        low: 5
        high: 20
```

최적화 실행:
```bash
uv run optimize.py
```
이 멍령어는 Sharpe Ratio(샤프 지수)를 최대화하는 최적의 `my_param` 값을 찾아줍니다.

## 성과 지표 (Performance Metrics)

시스템은 전략 성능 평가를 위해 다음과 같은 다양한 지표를 계산합니다:

| 지표 (Metric) | 설명 (Description) |
| :--- | :--- |
| **Total Return** | 포트폴리오의 총 수익률 (%) 입니다. |
| **Benchmark Return** | 첫 번째 심볼을 단순 보유(Buy & Hold)했을 때의 수익률입니다. |
| **CAGR** | 연평균 성장률 (Compound Annual Growth Rate) 입니다. |
| **Ann. Volatility** | 연환산 변동성(리스크)입니다. 수익률의 표준편차를 연율화한 것입니다. |
| **Sharpe Ratio** | 샤프 지수 (수익률 / 변동성). 높을수록 리스크 대비 수익이 좋습니다. |
| **Sortino Ratio** | 소르티노 지수. 하락 변동성(손실 위험)만을 리스크로 간주하여 평가합니다. |
| **Calmar Ratio** | 칼마 지수 (CAGR / 최대 낙폭). 낙폭 대비 성장률을 나타냅니다. |
| **Max Drawdown** | 최대 낙폭 (MDD). 고점 대비 가장 크게 하락한 비율입니다. |
| **DD Duration** | 최대 낙폭 상태가 지속된 기간(회복 기간 포함)입니다. |
| **Alpha** | 벤치마크 대비 초과 수익률입니다. |
| **Beta** | 벤치마크에 대한 민감도입니다. 1보다 크면 시장보다 변동성이 큽니다. |
| **Information Ratio** | 정보 비율 (Active Return / Tracking Error). 벤치마크 대비 얼마나 일관되게 초과 수익을 냈는지 측정합니다. |
| **Daily Win Rate** | 일별 승률. 전체 거래일 중 수익이 난 날의 비율입니다. |

결과는 `equity.csv` (수익 곡선) 및 `trades.csv` (거래 내역) 파일로 저장됩니다.
