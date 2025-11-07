# main.py
import io
import os
import math
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- Config ----------
WEIGHTS_PATH = "best_residual.h5"   # your model filename
WINDOW = 50
# ----------------------------

# SciPy for exponential fallback
USE_SCIPY = True
try:
    from scipy.optimize import curve_fit
except:
    USE_SCIPY = False

# TensorFlow / Keras
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except:
    TF_AVAILABLE = False

# ---------- FastAPI ----------
app = FastAPI(title="BMS Hybrid Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all (frontend can call us)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionPoint(BaseModel):
    cycle: float
    capacity: float

class PredictResponse(BaseModel):
    meta: Dict[str, Any]
    historical: List[PredictionPoint]
    predictions: List[PredictionPoint]

# ---------- helpers ----------
def read_table(upload: UploadFile) -> pd.DataFrame:
    raw = upload.file.read()
    name = upload.filename.lower()

    # CSV
    if name.endswith(".csv") or b"," in raw[:200]:
        return pd.read_csv(io.BytesIO(raw))

    # XLSX
    try:
        import openpyxl
        return pd.read_excel(io.BytesIO(raw))
    except:
        raise HTTPException(400, "Could not read CSV or Excel file")

def find_col(df, keys):
    for c in df.columns:
        for k in keys:
            if k in str(c).lower():
                return c
    return None

def fit_exponential(x, y):
    if not USE_SCIPY:
        return None
    try:
        a0 = max(y) - min(y)
        b0 = 1e-3
        c0 = min(y)
        popt, _ = curve_fit(lambda t,a,b,c: a*np.exp(-b*t)+c, x, y, p0=(a0,b0,c0), maxfev=20000)
        return popt
    except:
        return None

def exp_predict(x, params):
    a,b,c = params
    return a*np.exp(-b*x)+c

def fit_poly2(x, y):
    return np.polyfit(x,y,2)

def poly2_predict(x, p):
    return p[0]*x*x + p[1]*x + p[2]

def infer_step(cycles):
    diffs = np.diff(np.unique(cycles))
    return float(np.median(diffs)) if len(diffs)>0 else 1.0

# ---------- ROUTES ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "tf": TF_AVAILABLE,
        "weights_found": os.path.exists(WEIGHTS_PATH)
    }

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(file: UploadFile = File(...), future_cycles: int = Form(...)):

    if not TF_AVAILABLE:
        raise HTTPException(500, "TensorFlow not available")

    if not os.path.exists(WEIGHTS_PATH):
        raise HTTPException(500, "weights .h5 not found on server")

    df = read_table(file)

    cyc_col = find_col(df, ["cycle"])
    cap_col = find_col(df, ["capacity"])

    if cyc_col is None or cap_col is None:
        raise HTTPException(400, "Missing cycle/capacity column")

    df = df[[cyc_col, cap_col]].dropna().sort_values(cyc_col)
    df.columns = ["Cycle","Capacity"]

    cycles = df["Cycle"].to_numpy(float)
    caps   = df["Capacity"].to_numpy(float)

    # build trend
    x0 = cycles - cycles.min()
    params = fit_exponential(x0, caps)
    if params is None:
        p = fit_poly2(cycles, caps)
        trend = lambda x: poly2_predict(x, p)
        trend_name = "poly2"
    else:
        trend = lambda x: exp_predict(x - cycles.min(), params)
        trend_name = "exp"

    # load model
    model = load_model(WEIGHTS_PATH)

    # residual history
    trend_hist = trend(cycles)
    residuals = caps - trend_hist

    # scale residuals 0–1
    rmin = residuals.min()
    rmax = residuals.max()
    def scale(r): return (r-rmin)/(rmax-rmin+1e-12)
    def unscale(s): return s*(rmax-rmin+1e-12)+rmin

    hist = list(scale(residuals))
    if len(hist) < WINDOW:
        hist = [hist[0]]*(WINDOW-len(hist)) + hist

    # make future axis
    step = infer_step(cycles)
    last = cycles[-1]
    fut = np.arange(last+step, last+step*(future_cycles+1), step)

    preds = []
    for c in fut:
        window = np.array(hist[-WINDOW:], dtype=float).reshape(1, WINDOW, 1)
        rhat_s = float(model.predict(window, verbose=0)[0,0])
        rhat   = unscale(rhat_s)
        cap_p  = float(trend(np.array([c]))[0]) + rhat
        preds.append(cap_p)
        hist.append(rhat_s)

    return PredictResponse(
        meta={"engine":"hybrid","trend":trend_name,"window":WINDOW},
        historical=[PredictionPoint(cycle=float(c), capacity=float(y)) for c,y in zip(cycles,caps)],
        predictions=[PredictionPoint(cycle=float(c), capacity=float(y)) for c,y in zip(fut,preds)]
    )
