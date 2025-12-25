"""
=================================
XGBoost 房價預測最終部署穩定版
=================================
"""

import os
import json
from datetime import datetime

import joblib
import pandas as pd
import shap
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Noto Sans CJK TC"
mpl.rcParams["axes.unicode_minus"] = False


class HousePricePredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)

        self.model = joblib.load(os.path.join(base_dir, "model.pkl"))
        self.model_features = joblib.load(
            os.path.join(base_dir, "model_features.pkl")
        )

        # ⭐ 統一 SHAP 介面（比 TreeExplainer 穩）
        self.explainer = shap.Explainer(self.model)

        self.categorical_cols = ["district", "building_type", "main_use"]

    # --------------------------------------------------
    def _preprocess(self, case_dict: dict) -> pd.DataFrame:
        # 容錯：補齊 categorical
        for c in self.categorical_cols:
            case_dict.setdefault(c, "")

        df = pd.DataFrame([case_dict])

        df = pd.get_dummies(
            df,
            columns=self.categorical_cols,
            drop_first=False,
        )

        # ⭐ 嚴格對齊訓練特徵（你第一版的穩定核心）
        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0

        return df[self.model_features]

    # --------------------------------------------------
    def predict(self, case_dict: dict) -> dict:
        X = self._preprocess(case_dict)

        pred = float(self.model.predict(X)[0])
        explanation = self.explainer(X)

        # ===== SHAP Top 5 bar =====
        shap_vals = explanation.values[0]
        abs_vals = np.abs(shap_vals)
        idx = np.argsort(abs_vals)[-5:][::-1]

        fig_bar, ax = plt.subplots(figsize=(7, 4))
        ax.barh(
            X.columns[idx],
            abs_vals[idx],
        )
        ax.set_title("影響房價最大的因素（Top 5）")
        ax.invert_yaxis()

        # ===== SHAP waterfall =====
        fig_waterfall = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(explanation[0], show=False)

        # ===== 中文說明 =====
        texts = []
        for i in idx:
            direction = "提高" if shap_vals[i] > 0 else "降低"
            texts.append(
                f"• {X.columns[i]} 對價格有明顯{direction}影響"
            )

        return {
            "predicted_price": pred,
            "shap_bar_fig": fig_bar,
            "shap_waterfall_fig": fig_waterfall,
            "explanation": "\n".join(texts),
        }

    # --------------------------------------------------
    def export_bundle(self, case_dict, output_root="output"):
        result = self.predict(case_dict)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(output_root, f"prediction_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "prediction.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input": case_dict,
                    "predicted_price": round(result["predicted_price"], 2),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(os.path.join(out_dir, "explanation.txt"), "w", encoding="utf-8") as f:
            f.write(result["explanation"])

        result["shap_bar_fig"].savefig(
            os.path.join(out_dir, "shap_bar.png"), dpi=150, bbox_inches="tight"
        )
        result["shap_waterfall_fig"].savefig(
            os.path.join(out_dir, "shap_waterfall.png"),
            dpi=150,
            bbox_inches="tight",
        )

        plt.close("all")
        return out_dir
