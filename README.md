# P.L.A.N.T.S.

A serious game framework for daylight-driven indoor farming design with AI-based approximate DLI estimation.

**Paper #516** — eCAADe 2026

## Overview

This repo contains the data pipeline and ML models that power the Unity-based serious game. Users adjust building parameters (orientation, window ratio, tree presence, etc.) and get real-time GHR/DLI predictions for indoor farming viability.

## Structure

```
datasets/           Raw Excel simulation data (3 buildings, 4 conditions, 365 days)
datasets/processed/ Generated train/test splits (run preprocessing to create)
models/             Trained model files (.pkl)
models/unity_export/  ONNX models + config for Unity integration
notebooks/
  preprocessing.ipynb      Data loading, encoding, feature engineering, train/test split
  train/
    01_tree_models.ipynb             Random Forest, XGBoost, LightGBM training
    03_comparison_and_export.ipynb   Model comparison, ONNX export, Unity inference demo
combine_datasets.py   Merges raw Excel files into a single CSV with season labels
```

## Quickstart

```bash
pip install pandas openpyxl scikit-learn xgboost lightgbm onnxruntime onnxmltools seaborn joblib

# 1. Combine raw data
python combine_datasets.py

# 2. Run preprocessing notebook
jupyter notebook notebooks/preprocessing.ipynb

# 3. Train models
jupyter notebook notebooks/train/01_tree_models.ipynb

# 4. Compare & export to ONNX
jupyter notebook notebooks/train/03_comparison_and_export.ipynb
```

## Results

| Model | GHR R² | DLI R² | Inference |
|-------|--------|--------|-----------|
| **XGBoost** | **0.9931** | **0.9930** | **3.5ms** |
| LightGBM | 0.9723 | 0.9722 | 0.9ms |
| Random Forest | 0.9450 | 0.9450 | 16.4ms |

Best model (XGBoost) is exported as ONNX for Unity Barracuda.

## Unity Integration

The `models/unity_export/` folder contains:
- `daylight_ghr.onnx` / `daylight_dli.onnx` — prediction models
- `unity_config.json` — feature order, one-hot encoding maps, parameter definitions

User-adjustable parameters: Day of Year, Building, Orientation, Level, WWR, Window Transmittance, Tree Width, Tree Presence.

## License

See [LICENSE](LICENSE).
