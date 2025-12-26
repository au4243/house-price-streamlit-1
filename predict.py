# predict.py
import os
import joblib
import pandas as pd
import numpy as np


class HousePricePredictor:
    """
    房價預測器（Streamlit Cloud 安全版）
    - 不使用 matplotlib
    - 不顯示任何圖
    - 不處理中文字型
    - 只輸出「預測結果 + 文字解說」
    """

    def __init__(
        self,
        model_path="model.pkl",
        feature_path="model_features.pkl",
    ):
        # =========================
        # 檔案存在檢查
        # =========================
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"❌ 找不到特徵檔：{feature_path}")

        # =========================
        # 載入模型與特徵
        # =========================
        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

    # =========================
    # 特徵對齊（非常重要）
    # =========================
    def _align_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        # 缺的欄位補 0
        for col in self.model_features:
            if col not in df.columns:
                df[col] = 0

        # 多的欄位刪掉
        df = df[self.model_features]

        return df

    # =========================
    # 預測主流程
    # =========================
    def predict(self, input_data: dict) -> dict:
        """
        input_data: dict（來自 Streamlit 表單）
        """

        # 轉成 DataFrame
        input_df = pd.DataFrame([input_data])

        # 特徵對齊
        X = self._align_features(input_df)

        # 預測
        pred_price = float(self.model.predict(X)[0])

        # 文字版特徵影響說明
        explanation = self._text_feature_importance(X)

        return {
            "predicted_price": round(pred_price, 2),
            "explanation": explanation,
        }

    # =========================
    # 文字版特徵重要性
    # =========================
    def _text_feature_importance(self, X: pd.DataFrame, top_n: int = 5) -> list:
        """
        回傳文字解說（不畫圖、不用 shap）
        """

        explanations = []

        # Tree-based model（XGBoost / RandomForest）
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            importance_df = pd.DataFrame(
                {
                    "feature": self.model_features,
                    "importance": importances,
                    "value": X.iloc[0].values,
                }
            )

            importance_df = importance_df.sort_values(
                by="importance", ascending=False
            ).head(top_n)

            for _, row in importance_df.iterrows():
                explanations.append(
                    f"{row['feature']} 對預測有明顯影響（權重 {row['importance']:.3f}，輸入值 {row['value']}）"
                )

        else:
            explanations.append("此模型不支援特徵重要性分析")

        return explanations
