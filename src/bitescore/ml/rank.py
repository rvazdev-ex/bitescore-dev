from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import numpy as np

FEATURE_EXCLUDE = {"id"}

def default_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

def _feature_matrix(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    X = df[cols].values
    return X, cols

def rank_sequences(features_df: pd.DataFrame, model_path: str | None, train_demo: bool, outdir: Path):
    model = None
    if model_path and Path(model_path).exists():
        model = joblib.load(model_path)

    X, cols = _feature_matrix(features_df)

    if model is None:
        model = default_model()
        if train_demo:
            y = (
                0.6*features_df.get("aa_essential_frac", 0) +
                0.2*features_df.get("trypsin_K_sites", 0) +
                0.2*features_df.get("trypsin_R_sites", 0)
            ).values
        else:
            y = np.zeros(len(features_df))
        model.fit(X, y)
        mp = outdir / "model.joblib"
        joblib.dump(model, mp)
        model_path = str(mp)

    scores = model.predict(X)
    ranked = features_df.copy()
    ranked["digestibility_score"] = scores
    ranked = ranked.sort_values("digestibility_score", ascending=False).reset_index(drop=True)
    return ranked, model_path
