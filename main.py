# main.py
import io, os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler

# ---------- Config ----------
ONNX_PATH = os.environ.get("BMS_ONNX_PATH", "best_residual.onnx")
WINDOW = 50
ALLOW_TREND_ONLY_IF_NO_MODEL = False  # set True if you want fallback without ONNX
# ----------------------------

# Optional SciPy exponential fit (fallback to poly2 if not available)
USE_SCIPY = True
try:
    from scipy.optimize import curve_fit
except Exception:
    USE_SCIPY = False

app = FastAPI(title="BMS Hybrid Predictor (ONNX)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Schemas -----
class Point(BaseModel):
    cycle: float
    capacity: float

class PredictResponse(BaseModel):
    meta: Dict[str, Any]
    historical: List[Point]
    predictions: List[Point]

# ----- Helpers -----
def read_table(upload: UploadFile) -> pd.DataFrame:
    raw = upload.file.read()
    name = (upload.filename or "").lower()
    if not raw:
        raise HTTPException(400, "Empty file")
    # quick CSV sniff
    if name.endswith(".csv") or b"," in raw[:256]:
        try:
            return pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(400, f"Failed to read CSV: {e}")
    # excel
    try:
        import openpyxl  # noqa
        return pd.read_excel(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Failed to read Excel: {e}")

def find_col(df: pd.DataFrame, keys) -> Optional[str]:
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in keys):
            return c
    return None

def fit_exponential(x, y):
    if not USE_SCIPY:
        return None
    try:
        a0 = float(np.max(y) - np.min(y))
        b0 = 1e-3
        c0 = float(np.min(y))
        def f(t, a, b, c): return a * np.exp(-b * t) + c
        popt, _ = curve_fit(f, x, y, p0=(a0, b0, c0), maxfev=20000)
        return popt
    except Exception:
        return None

def exp_predict(x, params):
    a, b, c = params
    return a * np.exp(-b * x) + c

def fit_poly2(x, y):
    return np.polyfit(x, y, 2)

def poly2_predict(x, p):
    return p[0]*x**2 + p[1]*x + p[2]

def build_trend(cycles: np.ndarray, caps: np.ndarray):
    # stabilize exponent by shifting to start at 0
    offset = float(cycles.min())
    x0 = cycles - offset
    params = fit_exponential(x0, caps)
    if params is not None:
        def f(x: np.ndarray):
            return exp_predict(np.asarray(x, float) - offset, params)
        return f, "exp"
    # fallback: poly2
    p = fit_poly2(cycles, caps)
    def f(x: np.ndarray):
        x = np.asarray(x, float)
        return poly2_predict(x, p)
    return f, "poly2"

def infer_step(cycles: np.ndarray) -> float:
    diffs = np.diff(np.unique(cycles))
    return float(np.median(diffs)) if len(diffs) else 1.0

def ensure_model() -> Optional[ort.InferenceSession]:
    if not os.path.exists(ONNX_PATH):
        return None
    try:
        # input name is "input" because we exported with that signature
        return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise HTTPException(500, f"Failed to load ONNX: {e}")

# ----- Routes -----
@app.get("/health")
def health():
    return {
        "ok": True,
        "onnx_found": os.path.exists(ONNX_PATH),
        "window": WINDOW
    }

@app.post("/predict", response_model=PredictResponse)
def predict(
    file: UploadFile = File(...),
    future_cycles: int = Form(...)
):
    if future_cycles <= 0:
        raise HTTPException(400, "future_cycles must be > 0")

    df = read_table(file)

    cyc_col = find_col(df, ["cycle", "cycle count", "cycles"])
    cap_col = find_col(df, ["capacity"])
    efc_col = find_col(df, ["efc"])  # optional

    if cyc_col is None or cap_col is None:
        raise HTTPException(400, "Columns required: Cycle, Capacity (EFC optional).")

    cols = [cyc_col, cap_col] + ([efc_col] if efc_col else [])
    df = df[cols].copy().dropna().sort_values(cyc_col)
    df.columns = ["Cycle", "Capacity"] + (["EFC"] if efc_col else [])
    if len(df) < max(5, WINDOW):
        raise HTTPException(400, f"Need at least {max(5, WINDOW)} rows, got {len(df)}")

    cycles = df["Cycle"].to_numpy(float)
    caps   = df["Capacity"].to_numpy(float)
    efc    = df["EFC"].to_numpy(float) if "EFC" in df.columns else np.zeros_like(cycles)

    # 1) Trend on all history
    trend_func, trend_name = build_trend(cycles, caps)
    trend_hist = trend_func(cycles)
    resid_hist = caps - trend_hist

    # 2) Fit scaler on history features [Cycle, EFC, Residual]
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(np.stack([cycles, efc, resid_hist], axis=1))
    cmin, cmax = feat_scaler.data_min_[0], feat_scaler.data_max_[0]
    emin, emax = feat_scaler.data_min_[1], feat_scaler.data_max_[1]
    rmin, rmax = feat_scaler.data_min_[2], feat_scaler.data_max_[2]

    def scale_cycle(c):
        return (c - cmin) / (cmax - cmin + 1e-12)
    def scale_efc(e):
        return (e - emin) / (emax - emin + 1e-12)
    def scale_res(r):
        return (r - rmin) / (rmax - rmin + 1e-12)
    def unscale_res(s):
        return s * (rmax - rmin + 1e-12) + rmin

    # 3) Seed last WINDOW rows as [Cycle_raw, Cycle_s, EFC_s, Residual_s]
    seed_idx = np.arange(max(0, len(df)-WINDOW), len(df))
    seed_cyc = cycles[seed_idx]
    seed_efc = efc[seed_idx]
    seed_tr  = trend_hist[seed_idx]
    seed_res = caps[seed_idx] - seed_tr

    if len(seed_idx) < WINDOW:
        # left-pad using earliest values to reach WINDOW
        need = WINDOW - len(seed_idx)
        pad_c = np.full(need, cycles[0])
        pad_e = np.full(need, efc[0] if len(efc) else 0.0)
        pad_r = np.full(need, seed_res[0])
        seed_cyc = np.concatenate([pad_c, seed_cyc])
        seed_efc = np.concatenate([pad_e, seed_efc])
        seed_res = np.concatenate([pad_r, seed_res])

    hist_rows = []
    for c0, e0, r0 in zip(seed_cyc, seed_efc, seed_res):
        hist_rows.append([float(c0), float(scale_cycle(c0)), float(scale_efc(e0)), float(scale_res(r0))])

    # 4) Future cycle axis (keep step from history)
    step = infer_step(cycles)
    last = float(cycles[-1])
    future_axis = np.arange(last + step, last + step*(future_cycles+1), step, dtype=float)

    # 5) Inference
    session = ensure_model()
    if session is None:
        if ALLOW_TREND_ONLY_IF_NO_MODEL:
            y_future = trend_func(future_axis)
            preds = [Point(cycle=float(c), capacity=float(v)) for c, v in zip(future_axis, y_future)]
            engine = "trend-only"
            return PredictResponse(
                meta={"engine": engine, "trend": trend_name, "window": WINDOW, "note": "ONNX not found"},
                historical=[Point(cycle=float(c), capacity=float(y)) for c, y in zip(cycles, caps)],
                predictions=preds
            )
        else:
            raise HTTPException(503, "ONNX model not found on server.")

    input_name = session.get_inputs()[0].name  # "input" from export
    preds_out: List[Point] = []

    # simple EFC hold for future (you can improve later)
    efc_future_const = float(efc[-1]) if len(efc) else 0.0

    for c in future_axis:
        # build input window (WINDOW, 4)
        window = np.array(hist_rows[-WINDOW:], dtype=np.float32)  # shape (50,4)
        x = window.reshape(1, WINDOW, 4)  # (1,50,4)

        # model predicts residual_scaled
        rhat_scaled = float(session.run(None, {input_name: x})[0][0][0])
        rhat = unscale_res(rhat_scaled)

        cap_pred = float(trend_func(np.array([c]))[0]) + rhat
        preds_out.append(Point(cycle=float(c), capacity=cap_pred))

        # push to history for next step
        cyc_s = scale_cycle(c)
        efc_s = scale_efc(efc_future_const)
        hist_rows.append([float(c), float(cyc_s), float(efc_s), float(rhat_scaled)])

    return PredictResponse(
        meta={"engine":"hybrid-residual-onnx", "trend": trend_name, "window": WINDOW},
        historical=[Point(cycle=float(c), capacity=float(y)) for c, y in zip(cycles, caps)],
        predictions=preds_out
    )
