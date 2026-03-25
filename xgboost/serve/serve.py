import xgboost as xgb
import lightgbm as lgb
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
app = FastAPI()

xgb_model = xgb.Booster()
xgb_model.load_model("xgb_model.json")
lgb_model = lgb.Booster(model_file="lgb_model.txt")

class Features(BaseModel):
    values: List[float]  # 39 features in order: int1..int13, cat1..cat26
    
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: Features):
    num_cols = [f"int{i}" for i in range(1, 14)]
    cat_cols = [f"cat{i}" for i in range(1, 27)]
    col_names = num_cols + cat_cols

    
    x = pd.DataFrame([req.values], columns=col_names)
    xgb_prob = float(xgb_model.predict(xgb.DMatrix(x))[0])
    lgb_prob = float(lgb_model.predict(x)[0])
    return {
        "xgb_click_probability": xgb_prob,
        "lgb_click_probability": lgb_prob,
    }