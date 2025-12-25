import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Noto Sans CJK TC"
mpl.rcParams["axes.unicode_minus"] = False


class HousePricePredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)

        model_path = os.path.join(base_dir, "model.pkl")
        feature_path = os.path.join(base_dir, "model_features.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"❌ 找不到 model_features.pkl，請確認已 push 到 GitHub"
            )

        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

        self.explainer = shap.TreeExplainer(self.model)

    def _align_features(self, case_dict):
        df = pd.DataFrame([case_dict])
        df = pd.get_dummies(df)

        # 補齊訓練時欄位
        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0

        return df[self.model_features]

    def predict(self, case_dict):
        X = self._align_features(case_dict)

        pred = self.model.predict(X)[0]
        shap_values = self.explainer(X)

        # SHAP bar
        vals = np.abs(shap_values.values[0])
        idx = np.argsort(vals)[-5:][::-1]

        fig_bar, ax = plt.subplots(figsize=(7, 4))
        ax.barh(X.columns[idx], vals[idx])
        ax.set_title("影響房價最大的因素（Top 5）")
        ax.invert_yaxis()

        # SHAP waterfall
        fig_waterfall = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], show=False)

        # 中文說明
        explanation = []
        for i in idx:
            direction = "提高" if shap_values.values[0][i] > 0 else "降低"
            explanation.append(f"• {X.columns[i]} 對價格有明顯{direction}影響")

        return {
            "predicted_price": float(pred),
            "shap_bar_fig": fig_bar,
            "shap_waterfall_fig": fig_waterfall,
            "explanation": "\n".join(explanation),
        }
