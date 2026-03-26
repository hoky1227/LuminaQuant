# 최소 설치 프로필

LuminaQuant는 사용 목적에 맞춘 extras 기반 최소 설치를 지원합니다.

## 권장 프로필

### 백테스트 전용
```bash
uv sync --extra backtest --extra dev
```

### 최적화
```bash
uv sync --extra backtest --extra optimize --extra dev
```

### 바이낸스 라이브
```bash
uv sync --extra live-binance --extra dev
```

### MT5 라이브
```bash
uv sync --extra live-mt5 --extra dev
```

### Polymarket 라이브 (Phase 1)
```bash
uv sync --extra live-polymarket --extra dev
```
이 경로는 market data, paper/shadow workflow, 그리고 Polymarket 자격증명/개인키 설정 후 `allow_real_execution`을 명시적으로 켠 경우의 실험적 real execution 경로까지 포함합니다.

### 대시보드
```bash
uv sync --extra dashboard --extra dev
cd apps/dashboard_web && npm install
```
Python dashboard helper와 기본 Next.js 대시보드 런타임(Node 20+)이 필요한 경우 사용합니다.

### 전체 개발 설치
```bash
uv sync --extra backtest --extra optimize --extra live-binance --extra live-mt5 --extra live-polymarket --extra dashboard --extra dev
cd apps/dashboard_web && npm install
```

## 호환 alias
- `live`: 기존 live 설치 경로를 위한 호환 alias
- `all`: 폭넓은 로컬 개발용 편의 alias
