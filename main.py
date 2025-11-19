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

def classify_avg_efc(avg_efc: Optional[float]) -> Optional[Dict[str, Any]]:
    if avg_efc is None:
        return None
    if avg_efc < 1.5:
        return {
            "band": "low",
            "range": "< 1.5",
            "color": "blue",
            "advice": "Your charging pattern involves small partial cycles. Increase the charge window and keep SOC between 20%-80%."
        }
    if avg_efc <= 1.8:
        return {
            "band": "optimal",
            "range": "1.5 - 1.8",
            "color": "green",
            "advice": "Charging behavior is healthy. Keep the battery between 20%-80% for best life."
        }
    return {
        "band": "high",
        "range": "> 1.8",
        "color": "red",
        "advice": "Battery is seeing deep discharge cycles. Avoid letting SOC fall below 10%-20%."
    }


def classify_drop_per_100(drop_pct: Optional[float]) -> Optional[Dict[str, Any]]:
    if drop_pct is None:
        return None
    if drop_pct < 1:
        return {"band": "<1%", "advice": "Excellent battery health."}
    if drop_pct <= 2:
        return {"band": "1-2%", "advice": "Normal degradation."}
    if drop_pct <= 4:
        return {"band": "2-4%", "advice": "High degradation. Modify usage."}
    return {"band": ">4%", "advice": "Severe degradation. Battery under heavy stress."}


def combined_behavior_advice(avg_efc: Optional[float], drop_pct: Optional[float]) -> Optional[Dict[str, Any]]:
    if avg_efc is None or drop_pct is None:
        return None

    def efc_key(val: float) -> str:
        if val < 1.5:
            return "low"
        if val <= 1.8:
            return "good"
        return "high"

    def drop_key(val: float) -> str:
        if val < 1:
            return "<1"
        if val <= 2:
            return "1-2"
        return ">2"

    table = {
        "good": {
            "<1": "Excellent battery habits. Keep using a 20%-80% charging window.",
            "1-2": "Charging is good and aging is normal. Continue 20%-80% charging and avoid heat.",
            ">2": "Charging pattern is good, but degradation is high. Avoid fast charging and charging when hot."
        },
        "low": {
            "<1": "Degradation is low but avoid frequent small top-ups.",
            "1-2": "Increase your charging window. Avoid tiny 60%-70% charges; use larger ranges.",
            ">2": "Large drop with low EFC. Stop frequent small charges and stick to a stable 20%-80% window."
        },
        "high": {
            "<1": "Avoid letting the battery fall below 20%, even if drop looks small.",
            "1-2": "Charge earlier. Avoid discharging below 10%-20% to prevent rapid degradation.",
            ">2": "Deep cycles are causing major degradation. Avoid <20% SOC and 0-100% swings."
        }
    }

    return {
        "efc_band": efc_key(avg_efc),
        "drop_band": drop_key(drop_pct),
        "advice": table[efc_key(avg_efc)][drop_key(drop_pct)]
    }


def summarize_cycle_metrics(df: pd.DataFrame, group_size: int = 100) -> Dict[str, Any]:
    """
    Return average EFC plus overall capacity drop information.
    Capacity drop = ((first cycle capacity - last cycle capacity) / first cycle capacity) * 100.
    The per-100-cycle drop normalizes the total percentage drop by dividing by
    (total_cycles / 100) rounded to the nearest integer (minimum 1).
    """
    if "Cycle" not in df.columns or "Capacity" not in df.columns:
        raise ValueError("DataFrame must contain 'Cycle' and 'Capacity' columns.")

    ordered = df.sort_values("Cycle").reset_index(drop=True)
    avg_efc = None
    if "EFC" in ordered.columns and not ordered["EFC"].empty:
        avg_efc = float(ordered["EFC"].astype(float).mean())

    capacities = ordered["Capacity"].to_numpy(float)
    total_cycles = len(ordered)
    total_drop_pct = None
    per_100_drop_pct = None
    if total_cycles >= 2:
        first_cap = float(capacities[0])
        last_cap = float(capacities[-1])
        if first_cap != 0:
            total_drop_pct = float((first_cap - last_cap) / first_cap * 100.0)
            split = max(1, round(total_cycles / group_size))
            per_100_drop_pct = float(total_drop_pct / split)

    avg_efc_data = avg_efc
    drop_data = per_100_drop_pct
    efc_insight = classify_avg_efc(avg_efc_data)
    drop_insight = classify_drop_per_100(drop_data)
    combined_insight = combined_behavior_advice(avg_efc_data, drop_data)

    advice_list: List[str] = []
    for info in (efc_insight, drop_insight, combined_insight):
        if info and info.get("advice"):
            advice_list.append(info["advice"])

    return {
        "avg_efc": avg_efc,
        "total_capacity_drop_pct": total_drop_pct,
        "capacity_drop_pct_per_100_cycles": per_100_drop_pct,
        "insights": advice_list
    }

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

    metrics = summarize_cycle_metrics(df)

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
                meta={
                    "engine": engine,
                    "trend": trend_name,
                    "window": WINDOW,
                    "note": "ONNX not found",
                    **metrics
                },
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
        meta={
            "engine":"hybrid-residual-onnx",
            "trend": trend_name,
            "window": WINDOW,
            **metrics
        },
        historical=[Point(cycle=float(c), capacity=float(y)) for c, y in zip(cycles, caps)],
        predictions=preds_out
    )
