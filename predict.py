import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# Matplotlib（Cloud-safe）
# =========================
mpl.rcParams["font.family"] = "Noto Sans CJK TC"
mpl.rcParams["axes.unicode_minus"] = False


class HousePricePredictor:
    def __init__(self):
        # ===== 載入模型 =====
        self.model = joblib.load("model.pkl")

        # ===== 載入特徵順序 =====
        with open("model_features.json", encoding="utf-8") as f:
            self.model_features = json.load(f)

        # ===== SHAP Explainer（只建一次）=====
        self.explainer = shap.TreeExplainer(self.model)

    # =========================
    # 特徵對齊（最重要）
    # =========================
    def _align_features(self, case_dict: dict) -> pd.DataFrame:
        df = pd.DataFrame([case_dict])

        # --- One-hot encoding（示意，依你模型實際調整） ---
        df = pd.get_dummies(df)

        # --- 補齊缺失欄位 ---
        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0

        # --- 只保留訓練欄位順序 ---
        df = df[self.model_features]

        return df

    # =========================
    # SHAP Bar（Top 5）
    # =========================
    def _plot_shap_bar(self, shap_values, X):
        shap_mean = np.abs(shap_values.values[0])
        feature_names = X.columns

        top_idx = np.argsort(shap_mean)[-5:][::-1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(
            feature_names[top_idx],
            shap_mean[top_idx]
        )
        ax.set_title("影響房價最大的因素（Top 5）")
        ax.invert_yaxis()

        return fig

    # =========================
    # SHAP Waterfall
    # =========================
    def _plot_shap_waterfall(self, shap_values):
        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title("單筆預測 SHAP 解釋")

        return fig

    # =========================
    # 中文說明生成
    # =========================
    def _generate_explanation(self, shap_values, X):
        sv = shap_values.values[0]
        features = X.columns

        idx = np.argsort(np.abs(sv))[-5:][::-1]

        lines = []
        for i in idx:
            direction = "提高" if sv[i] > 0 else "降低"
            lines.append(
                f"• {features[i]} 對價格有明顯{direction}影響"
            )

        return "\n".join(lines)

    # =========================
    # 對外主介面
    # =========================
    def predict(self, case_dict: dict) -> dict:
        X = self._align_features(case_dict)

        # ===== 預測 =====
        pred = self.model.predict(X)[0]

        # ===== SHAP =====
        shap_values = self.explainer(X)

        # ===== 圖表 =====
        shap_bar_fig = self._plot_shap_bar(shap_values, X)
        shap_waterfall_fig = self._plot_shap_waterfall(shap_values)

        # ===== 中文解釋 =====
        explanation = self._generate_explanation(shap_values, X)

        return {
            "predicted_price": float(pred),
            "explanation": explanation,
            "shap_bar_fig": shap_bar_fig,
            "shap_waterfall_fig": shap_waterfall_fig,
        }
