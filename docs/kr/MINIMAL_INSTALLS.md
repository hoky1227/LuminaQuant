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

### 대시보드
```bash
uv sync --extra dashboard --extra dev
```

### 전체 개발 설치
```bash
uv sync --extra backtest --extra optimize --extra live-binance --extra live-mt5 --extra live-polymarket --extra dashboard --extra dev
```

## 호환 alias
- `live`: 기존 live 설치 경로를 위한 호환 alias
- `all`: 폭넓은 로컬 개발용 편의 alias
