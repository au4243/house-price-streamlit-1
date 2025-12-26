import os
import joblib
import shap
import numpy as np
import pandas as pd


class HousePricePredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "model.pkl")
        feature_path = os.path.join(base_dir, "model_features.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"❌ 找不到 model_features.pkl，請確認已 push 到 GitHub")

        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

        # SHAP 解釋器
        self.explainer = shap.TreeExplainer(self.model)

    # =========================
    # 特徵對齊（關鍵）
    # =========================
    def _align_features(self, case_dict):
        df = pd.DataFrame([case_dict])
        df = pd.get_dummies(df)

        # 補齊訓練時的欄位
        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0

        return df[self.model_features]

    # =========================
    # 特徵轉中文人話
    # =========================
    def _feature_to_human(self, feature, value):
        # 類別型（one-hot）
        if feature.startswith("district_"):
            return f"位於「{feature.replace('district_', '')}」"
        if feature.startswith("building_type_"):
            return f"建物型態為「{feature.replace('building_type_', '')}」"
        if feature.startswith("main_use_"):
            return f"主要用途為「{feature.replace('main_use_', '')}」"

        # 數值 / 布林型
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
    # 預測主函式
    # =========================
    def predict(self, case_dict):
        # 特徵處理
        X = self._align_features(case_dict)

        # 模型預測
        pred = float(self.model.predict(X)[0])

        # SHAP 解釋
        shap_values = self.explainer(X)
        shap_vals = shap_values.values[0]
        feature_names = X.columns

        # 基準值（模型平均值）
        base_value = float(self.explainer.expected_value)

        explanation_lines = []
        cumulative_price = base_value  # 從基準值開始累加
        explanation_lines.append(f"📌 模型基準單價約為 {base_value:.1f} 萬 / 坪，以下條件使價格進行調整：")

        # Top 5 影響因素
        idx = np.argsort(np.abs(shap_vals))[-5:][::-1]

        for i in idx:
            feat = feature_names[i]
            shap_val = shap_vals[i]
            human_text = self._feature_to_human(feat, X.iloc[0][feat])
            cumulative_price += shap_val
            direction = "推升" if shap_val > 0 else "下修"
            explanation_lines.append(f"👉 {human_text}，使單價約 {direction} {abs(shap_val):.1f} 萬 / 坪")

        explanation_lines.append(f"\n➡️ 綜合以上因素後，模型推估本案合理單價約為 {cumulative_price:.1f} 萬 / 坪。")

        return {
            "predicted_price": pred,
            "explanation": "\n".join(explanation_lines),
        }

