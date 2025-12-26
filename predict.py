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
            raise FileNotFoundError(
                f"❌ 找不到 model_features.pkl，請確認已 push 到 GitHub"
            )

        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

        # SHAP 解釋器（XGBoost / Tree-based 專用）
        self.explainer = shap.TreeExplainer(self.model)

    # =========================
    # 特徵對齊（關鍵）
    # =========================
    def _align_features(self, case_dict):
        df = pd.DataFrame([case_dict])
        df = pd.get_dummies(df)

        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0

        return df[self.model_features]

    # =========================
    # 特徵翻成人話
    # =========================
    def _feature_to_human(self, feature, value):
        if feature.startswith("district_"):
            return f"位於「{feature.replace('district_', '')}」"

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
    # 預測主流程（含金額解釋）
    # =========================
    def predict(self, case_dict):
        X = self._align_features(case_dict)

        # 預測單價（萬 / 坪）
        pred_price = float(self.model.predict(X)[0])

        # SHAP 解釋
        shap_values = self.explainer(X)

        base_value = shap_values.base_values[0]
        shap_contribs = shap_values.values[0]

        # 取影響最大的前 5 項
        idx = np.argsort(np.abs(shap_contribs))[-5:][::-1]

        explanation = []

        explanation.append(
            f"📌 模型基準單價約為 **{base_value:.1f} 萬 / 坪**，"
            "以下條件使價格進行調整："
        )

        for i in idx:
            feature = X.columns[i]
            shap_val = shap_contribs[i]

            direction = "推升" if shap_val > 0 else "下修"
            amount = abs(shap_val)

            human_text = self._feature_to_human(
                feature,
                X.iloc[0][feature]
            )

            explanation.append(
                f"• {human_text}，使單價約 **{direction} {amount:.1f} 萬 / 坪**。"
            )

        explanation.append(
            f"\n➡️ 綜合以上因素後，模型推估本案合理單價約為 "
            f"**{pred_price:.1f} 萬 / 坪**。"
        )

        return {
            "predicted_price": pred_price,
            "explanation": "\n".join(explanation),
        }
