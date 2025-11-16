import os
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """
    Payload de entrada para o endpoint de predição.

    `observation` é o vetor de estado que o modelo PPO espera.
    No notebook de treinamento, o espaço de observação tinha shape (11,):
      [cash] + [shares_owned (5)] + [prices (5)] -> 1 + 5 + 5 = 11.
    """

    observation: List[float]


class PredictionResponse(BaseModel):
    """
    Resposta do endpoint de predição.

    - raw_action: saída direta do modelo ONNX.
    - allocations: ação normalizada via softmax para virar pesos de portfólio.
    """

    raw_action: List[float]
    allocations: List[float]


TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INITIAL_BALANCE: float = 100000.0
TRANSACTION_COST: float = 0.001


def load_onnx_session() -> ort.InferenceSession:
    """
    Carrega o modelo ONNX de política PPO em CPU.
    """
    base_dir = Path(__file__).parent
    model_path = base_dir / "ppo_trader_onnx" / "ppo_policy_100k.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo ONNX não encontrado em {model_path}")

    # Providers somente CPU
    sess = ort.InferenceSession(
        model_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    return sess


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax numericamente estável para converter logits em pesos de portfólio.
    """
    x = x.astype(np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def fetch_price_data() -> tuple[pd.DataFrame, str]:
    """
    Baixa dados de preço via Alpha Vantage (TIME_SERIES_DAILY_ADJUSTED)
    de 2015 até hoje.

    Se Alpha Vantage falhar / vier vazio, usa um CSV local em
    data_fallback/sp500.csv. Se ainda assim falhar, gera uma série sintética
    como último recurso.

    Retorna:
        (dataframe_de_precos, data_source)
        data_source in {"alpha_vantage", "csv_fallback", "synthetic"}
    """
    # --- 1) Alpha Vantage, se houver API key ---
    # Aceita dois nomes de variável para conveniência:
    # - ALPHAVANTAGE_API_KEY (recomendado)
    # - ALPHA_KEY           (como configurado no Render)
    api_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_KEY")
    if api_key:
        try:
            import requests

            series_map: dict[str, pd.Series] = {}
            for ticker in TICKERS:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": ticker,
                    "outputsize": "full",
                    "apikey": api_key,
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data_json = resp.json()
                ts = data_json.get("Time Series (Daily)", {})
                if not ts:
                    print(f"[WARN] Alpha Vantage retornou vazio para {ticker}")
                    continue

                dates = []
                closes = []
                for date_str, daily in ts.items():
                    # usa o preço de fechamento ajustado ou normal
                    close_str = daily.get("5. adjusted close") or daily.get("4. close")
                    if close_str is None:
                        continue
                    dates.append(pd.to_datetime(date_str))
                    closes.append(float(close_str))

                if dates:
                    s = pd.Series(closes, index=pd.DatetimeIndex(dates)).sort_index()
                    # filtra de 2015 em diante
                    s = s[s.index >= pd.Timestamp("2015-01-01")]
                    series_map[ticker] = s

            if series_map:
                df_alpha = pd.DataFrame(series_map).dropna()
                if not df_alpha.empty:
                    return df_alpha, "alpha_vantage"
        except Exception as exc:  # pragma: no cover - proteção em produção
            print(f"[WARN] Falha ao baixar dados com Alpha Vantage: {exc}")

    # --- 2) Fallback local: CSV em data_fallback/sp500.csv ---
    try:
        base_dir = Path(__file__).parent
        csv_path = base_dir / "data_fallback" / "sp500.csv"
        if csv_path.exists():
            df_csv = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            # garante que só usamos as colunas dos tickers esperados
            available_cols = [c for c in df_csv.columns if c in TICKERS]
            if available_cols:
                close_csv = df_csv[available_cols].dropna()
                if not close_csv.empty:
                    return close_csv, "csv_fallback"
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Falha ao carregar CSV de fallback: {exc}")

    # --- 3) Último recurso: dados sintéticos (não recomendados em produção) ---
    num_days = 252  # ~1 ano útil
    dates = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq="B")

    rng = np.random.default_rng(seed=42)
    # preços iniciais aproximados
    start_prices = np.array([180.0, 400.0, 140.0, 160.0, 800.0], dtype=np.float64)
    # random walk multiplicativo
    returns = rng.normal(loc=0.0002, scale=0.02, size=(num_days, len(TICKERS)))
    prices = np.empty_like(returns)
    prices[0] = start_prices
    for i in range(1, num_days):
        prices[i] = prices[i - 1] * (1.0 + returns[i])

    close_synthetic = pd.DataFrame(prices, index=dates, columns=TICKERS)
    return close_synthetic, "synthetic"


app = FastAPI(
    title="Deep RL Trading Agent API",
    description=(
        "API REST para servir o modelo PPO exportado em ONNX "
        "e fornecer dados agregados para dashboards de trading."
    ),
    version="1.1.0",
)

onnx_session: ort.InferenceSession | None = None

# Configuração básica de CORS para permitir chamadas do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste para a URL do seu dashboard em produção, se quiser restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    """
    Carrega o modelo ONNX uma única vez na inicialização da API.
    """
    global onnx_session
    try:
        onnx_session = load_onnx_session()
    except Exception as exc:
        # Se falhar aqui, as requisições subsequentes retornarão erro 500,
        # mas a app ainda sobe, facilitando debug no Render.
        print(f"[ERRO] Falha ao carregar modelo ONNX: {exc}")


@app.get("/")
def read_root() -> dict:
    """
    Endpoint de saúde / informação básica da API.
    """
    return {
        "status": "ok",
        "message": "Deep RL Trading Agent API (PPO ONNX) rodando.",
        "model_loaded": onnx_session is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Gera uma ação a partir de uma observação do ambiente.

    Espera um vetor de 11 floats em `observation`:
      - índice 0: saldo em caixa (cash)
      - índices 1-5: quantidade de ações por ativo
      - índices 6-10: preços atuais dos 5 ativos
    """
    if onnx_session is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo ONNX não carregado. Verifique logs do servidor.",
        )

    obs = np.asarray(request.observation, dtype=np.float32)

    if obs.shape != (11,):
        raise HTTPException(
            status_code=400,
            detail=f"Esperado vetor de observação com shape (11,), "
            f"recebido shape {obs.shape}.",
        )

    # Prepara entrada batch (1, 11)
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    try:
        outputs = onnx_session.run(
            [output_name],
            {input_name: obs.reshape(1, -1)},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao executar inferência ONNX: {exc}",
        ) from exc

    raw_action = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    allocations = softmax(raw_action)

    return PredictionResponse(
        raw_action=raw_action.tolist(),
        allocations=allocations.tolist(),
    )


@app.get("/api/v1/dashboard-data")
def get_dashboard_data() -> dict:
    """
    Endpoint para alimentar um dashboard de trading de RL.

    - Baixa dados de fechamento de 1 ano para 5 tickers.
    - Calcula benchmark Buy & Hold (B&H).
    - Roda um backtest simples do agente PPO via modelo ONNX.
    """
    if onnx_session is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo ONNX não carregado. Verifique logs do servidor.",
        )

    # Tenta baixar dados; se falhar, cai em série sintética
    data, data_source = fetch_price_data()

    # --- Benchmark Buy & Hold ---
    first_prices = data.iloc[0].values.astype(np.float64)
    dollars_per_stock = INITIAL_BALANCE / len(TICKERS)
    shares_per_stock = dollars_per_stock / first_prices

    benchmark_history: List[float] = []
    for i in range(len(data)):
        current_prices = data.iloc[i].values.astype(np.float64)
        portfolio_value = float(np.dot(shares_per_stock, current_prices))
        benchmark_history.append(portfolio_value)

    # --- Simulação do Agente (Backtest) ---
    balance: float = INITIAL_BALANCE
    shares_owned = np.zeros(len(TICKERS), dtype=np.float64)
    portfolio_value: float = INITIAL_BALANCE
    agent_history: List[float] = [INITIAL_BALANCE]

    # input / output names do modelo
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    normalized_action_final = None

    for i in range(len(data) - 1):
        current_prices = data.iloc[i].values.astype(np.float64)

        # Estado: [cash] + [shares_owned] + [current_prices]
        obs = np.concatenate(
            [np.array([balance], dtype=np.float32), shares_owned.astype(np.float32), current_prices.astype(np.float32)]
        ).astype(np.float32)

        obs_batch = obs.reshape(1, -1)

        try:
            outputs = onnx_session.run(
                [output_name],
                {input_name: obs_batch},
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao executar inferência ONNX durante backtest: {exc}",
            ) from exc

        action_logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        action_exp = np.exp(action_logits - np.max(action_logits))
        normalized_action = action_exp / np.sum(action_exp)

        # Lógica de trade (replicando o step do ambiente)
        target_dollar_value = portfolio_value * normalized_action
        target_shares = target_dollar_value / current_prices
        shares_to_trade = target_shares - shares_owned

        trade_value = float(np.dot(shares_to_trade, current_prices))
        fees = TRANSACTION_COST * abs(trade_value)

        balance -= trade_value + fees
        shares_owned = target_shares

        # Valor no dia seguinte
        next_day_prices = data.iloc[i + 1].values.astype(np.float64)
        portfolio_value = float(balance + np.dot(shares_owned, next_day_prices))
        agent_history.append(portfolio_value)

        normalized_action_final = normalized_action

    # Caso, por algum motivo, o loop não rode
    if normalized_action_final is None:
        normalized_action_final = np.full(len(TICKERS), 1.0 / len(TICKERS), dtype=np.float64)

    current_allocation = {
        ticker: float(weight) for ticker, weight in zip(TICKERS, normalized_action_final)
    }

    price_history = data.reset_index().to_dict("records")

    return {
        "tickers": TICKERS,
        "data_source": data_source,
        "initial_balance": INITIAL_BALANCE,
        "transaction_cost": TRANSACTION_COST,
        "current_allocation": current_allocation,
        "agent_history": agent_history,
        "benchmark_history": benchmark_history,
        "price_history": price_history,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


