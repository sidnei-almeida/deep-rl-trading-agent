from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
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


app = FastAPI(
    title="Deep RL Trading Agent API",
    description=(
        "API REST para servir o modelo PPO exportado em ONNX "
        "a partir do Notebook 03 (Model Training and Evaluation)."
    ),
    version="1.0.0",
)

onnx_session: ort.InferenceSession | None = None


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


