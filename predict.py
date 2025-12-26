import os
import joblib
import shap
import numpy as np
import pandas as pd

# =========================
# Matplotlib 中文字型（內嵌，跨平台穩定）
# =========================
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

BASE_DIR = os.path.dirname(__file__)
FONT_PATH = os.path.join(
    BASE_DIR,
    "fonts",
    "NotoSansCJKtc-Regular.ttf"
)

if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(
        f"❌ 找不到中文字型檔，請確認存在：{FONT_PATH}"
    )

font_prop = font_manager.FontProperties(fname=FONT_PATH)

mpl.rcParams["font.family"] = font_prop.get_name()
mpl.rcParams["axes.unicode_minus"] = False


class HousePricePredictor:
    def __init__(self):
        # =========================
        # 載入模型與特徵
        # =========================
        model_path = os.path.join(BASE_DIR, "model.pkl")
        feature_path = os.path.join(BASE_DIR, "model_features.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"❌ 找不到 model_features.pkl，請確認已 push 到 GitHub"
            )

        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

        # SHAP 解釋器
        self.explainer = shap.TreeExplainer(self.model)

    # =========================
    # 特徵對齊（實務關鍵）
    # =========================
    def _align_features(self, case_dict):
        df = pd.DataFrame([case_dict])
        df = pd.get_dummies(df)

        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0

        return df[self.model_features]

    # =========================
    # 特徵轉中文人話
    # =========================
    def _feature_to_human(self, feature, value):
        if feature.startswith("district_"):
            return f"位於 {feature.replace('district_', '')}"

        if feature.startswith("building_type_"):
            return f"建物型態為「{feature.replace('building_type_', '')}」"

        if feature.startswith("main_use_"):
            return f"主要用途為「{feature.replace('main_use_', '')}」"

        HUMAN_MAP = {
            "main_area": f"主建物面積約 {value:.1f} 坪",
            "balcony_area": f"陽台面積約 {value:.1f} 坪",
            "building_age": f"屋齡約 {int(value)} 年",
            "floor": f"位於第 {int(value)} 樓",
            "total_floors": f"建物總樓層 {int(value)} 樓",
            "has_parking": "具備車位" if value == 1 else "未附車位",
            "has_elevator": "設有電梯" if value == 1 else "未設電梯",
        }

        return HUMAN_MAP.get(feature, feature)

    # =========================
    # 預測主流程
    # =========================
    def predict(self, case_dict):
        X = self._align_features(case_dict)

        # 預測價格
        pred = self.model.predict(X)[0]

        # SHAP 解釋
        shap_values = self.explainer(X)

        # =========================
        # SHAP Bar（Top 5）
        # =========================
        vals = np.abs(shap_values.values[0])
        idx = np.argsort(vals)[-5:][::-1]

        fig_bar, ax = plt.subplots(figsize=(7, 4))
        ax.barh(X.columns[idx], vals[idx])
        ax.set_title("影響房價最大的因素（Top 5）", fontproperties=font_prop)
        ax.invert_yaxis()

        # 確保座標軸也吃字型
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)

        # =========================
        # SHAP Waterfall
        # =========================
        fig_waterfall = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], show=False)

        # =========================
        # 中文估價說明
        # =========================
        explanation = []

        for i in idx:
            feature = X.columns[i]
            shap_val = shap_values.values[0][i]
            direction = "正向支撐" if shap_val > 0 else "負向影響"

            human_text = self._feature_to_human(
                feature,
                X.iloc[0][feature]
            )

            explanation.append(
                f"• {human_text}，對本案單價形成{direction}。"
            )

        return {
            "predicted_price": float(pred),
            "shap_bar_fig": fig_bar,
            "shap_waterfall_fig": fig_waterfall,
            "explanation": "\n".join(explanation),
        }
