## Deep RL Trading Agent API

Production-ready REST API for serving a **PPO-based reinforcement learning trading policy** exported to **ONNX**.  
The service is implemented with **FastAPI**, runs purely on **CPU**, and is fully prepared for deployment on **Render**.

---

## 1. Overview

This project exposes a trained **Proximal Policy Optimization (PPO)** policy as a web service.  
The model was trained in a custom `TradingEnv` (Gymnasium-style environment) using `stable-baselines3` and exported to ONNX in the training notebook (`03_Model_Training_and_Evaluation.ipynb`).

The API:
- **Loads** the ONNX policy at startup using `onnxruntime` (CPU only).
- **Accepts** a single observation vector representing the current trading state.
- **Returns** both:
  - the **raw policy output** (logits / unnormalized actions), and
  - a **softmax-normalized allocation vector** that can be interpreted as target portfolio weights.

This makes it suitable as a backend for:
- Live or paper trading systems,
- Strategy simulators,
- Research dashboards and monitoring tools.

---

## 2. Architecture

- **Language**: Python 3.11 (Render is configured with `PYTHON_VERSION=3.11.8`).
- **Framework**: FastAPI.
- **Model Runtime**: ONNX Runtime (CPU execution provider).
- **Web Server**: Uvicorn.
- **Model File**: `ppo_trader_onnx/ppo_policy_100k.onnx`.

High-level flow:

1. API starts and triggers the FastAPI `startup` event.
2. The ONNX model is loaded once into memory via `onnxruntime.InferenceSession`.
3. Each `/predict` call:
   - Validates and reshapes the input observation.
   - Performs ONNX inference.
   - Applies softmax to the raw action to compute portfolio allocations.

---

## 3. Observation & Action Space

The ONNX model represents the policy head of a PPO agent trained on a custom `TradingEnv`.  
The **observation vector** has shape **(11,)**:

- **Index 0**: cash balance (`cash`).
- **Indices 1–5**: number of shares owned for each of the 5 assets (`shares_owned`).
- **Indices 6–10**: current prices for each of the 5 assets (`prices`).

In the original environment:

- The action space is a continuous vector of shape `(5,)` representing per-asset logits.
- These logits are passed through **softmax** to obtain portfolio weights.

The ONNX policy follows this structure; the API applies softmax after inference so you receive:

- **`raw_action`**: model output logits.
- **`allocations`**: normalized weights (softmax) that sum to ~1.0.

---

## 4. Project Structure

- **`app.py`**  
  Main FastAPI application. Handles:
  - ONNX model loading (`onnxruntime.InferenceSession`),
  - Request/response validation (Pydantic models),
  - `/` and `/predict` endpoints.

- **`ppo_trader_onnx/ppo_policy_100k.onnx`**  
  ONNX-exported PPO policy (≈40 KB), exported from the 100k-timestep model in the training notebook.

- **`requirements.txt`**  
  Python dependencies for CPU-only inference and API server.

- **`render.yaml`**  
  Infrastructure-as-code file for deploying this service on **Render** as a Python web service.

- **`notebooks/`**  
  Jupyter notebooks for data processing and model training/evaluation (not required at runtime).

---

## 5. API Endpoints

### 5.1 Health Check

- **Method**: `GET`
- **Path**: `/`
- **Description**: Basic health and model status check.

**Response Example**

```json
{
  "status": "ok",
  "message": "Deep RL Trading Agent API (PPO ONNX) rodando.",
  "model_loaded": true
}
```

> `model_loaded = true` indicates the ONNX model was successfully loaded at startup.

---

### 5.2 Prediction

- **Method**: `POST`
- **Path**: `/predict`
- **Description**: Generates an action from a single observation of the trading environment.

#### Request Body

```json
{
  "observation": [100000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 3000.0, 150.0, 350.0, 700.0]
}
```

**Constraints:**

- `observation` must be:
  - a list of **11 numeric values (floats)**,
  - representing `[cash, 5x shares_owned, 5x prices]` in that exact order.

If the shape is not `(11,)`, the API returns HTTP **400 Bad Request** with an explicit error message.

#### Response Body

```json
{
  "raw_action": [
    -0.12345671653747559,
    0.541234016418457,
    0.012345678903341293,
    -0.7890123128890991,
    0.35999998450279236
  ],
  "allocations": [
    0.140123456716,
    0.322345678903,
    0.184567890123,
    0.065432109877,
    0.287530864381
  ]
}
```

- **`raw_action`**  
  Raw logits output by the policy head.

- **`allocations`**  
  Softmax-normalized weights derived from `raw_action`.  
  Interpretable as target portfolio weights for the 5 assets (usually sum to 1.0 up to numerical precision).

---

## 6. Local Development & Testing

### 6.1 Prerequisites

- **Python**: 3.11.x (recommended: 3.11.8 to match Render config).
- **Pip**: latest version.

### 6.2 Installation

From the project root:

```bash
pip install -r requirements.txt
```

### 6.3 Running the API Locally

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- `http://localhost:8000`
- `http://localhost:8000/docs` – interactive Swagger UI
- `http://localhost:8000/redoc` – alternative API docs

### 6.4 Example `curl` Call

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "observation": [100000, 0, 0, 0, 0, 0, 500, 3000, 150, 350, 700]
  }'
```

---

## 7. Deployment on Render

This repository includes a **Render** configuration file: `render.yaml`.

### 7.1 Service Definition (`render.yaml`)

- **Service Type**: `web`
- **Environment**: `python`
- **Plan**: `free` (can be adjusted to `starter`, `standard`, etc.).
- **Python Version**: `PYTHON_VERSION=3.11.8`
- **Build Command**:

  ```yaml
  buildCommand: "pip install -r requirements.txt"
  ```

- **Start Command**:

  ```yaml
  startCommand: "uvicorn app:app --host 0.0.0.0 --port $PORT"
  ```

### 7.2 Steps to Deploy

1. **Push the project to GitHub** (or another supported Git provider).
2. In the **Render Dashboard**:
   - Click **"New" → "Web Service"**.
   - Connect to the repository containing this project.
   - Render will automatically detect `render.yaml` and use it to configure the service.
3. Confirm settings and deploy.
4. After deployment, you can:
   - Check health via `GET /`.
   - Test predictions via `POST /predict` using the base URL generated by Render.

---

## 8. Environment & Dependencies

### 8.1 Python Version

- The service is configured for **Python 3.11.8** on Render.
- Local environments should use Python **3.11.x** to avoid version mismatch, especially for `onnxruntime`.

### 8.2 Key Dependencies

- **FastAPI** – high-performance web framework for building APIs.
- **Uvicorn** – ASGI server used to serve the FastAPI application.
- **ONNX Runtime** – model inference engine, CPU-only.
- **NumPy** – numerical operations (input reshaping, softmax).
- **Pydantic** – request and response schema validation.

All dependencies are pinned in `requirements.txt` for reproducible environments.

---

## 9. Security & Production Considerations

- **Input Validation**  
  The `/predict` endpoint validates the shape of the observation and returns HTTP 400 for invalid inputs.

- **Model Loading**  
  The ONNX model is loaded once during startup for efficiency. If loading fails, requests will receive an HTTP 500 error and logs should be inspected.

- **Authentication / Authorization**  
  The current implementation is **open by design** for simplicity.  
  For real-world production use, consider adding:
  - API keys, OAuth2, or JWT auth,
  - IP allowlists or private network deployment,
  - Rate limiting and logging.

- **Monitoring & Logging**  
  When deploying to Render or other platforms, make sure to:
  - Monitor response times,
  - Log failed inferences and input payloads (without leaking sensitive data),
  - Track model version and config for reproducibility.

---

## 10. Limitations & Next Steps

- **Model Behavior**  
  The current ONNX model is a snapshot of a PPO policy trained on historical data. Its live performance will depend on:
  - Market regime shifts,
  - Transaction costs and slippage,
  - Data feed quality and latency.

- **State Construction**  
  This API expects a **fully constructed observation vector**. In a production trading system, you may want to:
  - Build a dedicated state-construction layer (e.g., from positions and price feeds),
  - Validate that the observation strictly matches the training environment semantics.

- **Extensibility**  
  Possible extensions:
  - Add batch prediction endpoints.
  - Expose additional diagnostics (e.g., value function estimates).
  - Integrate risk controls (e.g., capping allocations or notional exposure).

If you plan to extend this API (e.g., different models, markets, or environments), the current structure provides a clean, modular starting point for a more comprehensive RL trading microservice.


