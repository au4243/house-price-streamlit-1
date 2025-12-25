"""
=================================
XGBoost 房價預測最終部署版模組（單檔完整版）
=================================
"""

import os
import json
from datetime import datetime

import joblib
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")  # ⭐ 關鍵：雲端 / 無 GUI 必備
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# Matplotlib 中文設定
# =========================
mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False


class HousePricePredictor:
    """房價預測與 SHAP 解釋模組（正式部署等級）"""

    def __init__(
        self,
        model_path: str = "model.pkl",
        feature_path: str = "model_features.pkl",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"❌ 找不到 model_features.pkl，請確認已 push 到 GitHub"
            )      
      
      
        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

        # ⭐ 統一使用新介面
        self.explainer = shap.Explainer(self.model)

        self.categorical_cols = [
            "district",
            "building_type",
            "main_use",
        ]

    # --------------------------------------------------
    @staticmethod
    def _pretty_name(col: str, value=None) -> str:
        if col.startswith("district_"):
            return f"行政區：{col.replace('district_', '')}"
        if col.startswith("building_type_"):
            return f"建物型態：{col.replace('building_type_', '')}"
        if col.startswith("main_use_"):
            return f"主要用途：{col.replace('main_use_', '')}"

        mapping = {
            "building_age": "屋齡（年）",
            "building_area_sqm": "建物移轉面積（㎡）",
            "main_area": "主建物面積（坪）",
            "balcony_area": "陽台面積（坪）",
            "floor": "所在樓層",
            "total_floors": "總樓層數",
            "has_parking": "是否有車位",
            "has_elevator": "是否有電梯",
        }
        name = mapping.get(col, col)
        return f"{name} = {value}" if value is not None else name

    # --------------------------------------------------
    def _preprocess(self, case_dict: dict) -> pd.DataFrame:
        # ⭐ 補齊缺失 categorical，避免 get_dummies 爆炸
        for c in self.categorical_cols:
            case_dict.setdefault(c, "")

        df = pd.DataFrame([case_dict])

        df = pd.get_dummies(
            df,
            columns=self.categorical_cols,
            drop_first=False,
        )

        missing_cols = set(self.model_features) - set(df.columns)
        if missing_cols:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(0, index=df.index, columns=list(missing_cols)),
                ],
                axis=1,
            )

        return df[self.model_features]

    # --------------------------------------------------
    def predict(self, case_dict: dict) -> float:
        X_case = self._preprocess(case_dict)
        return float(self.model.predict(X_case)[0])

    # --------------------------------------------------
    def shap_explain(self, case_dict: dict):
        X_case = self._preprocess(case_dict)
        explanation = self.explainer(X_case)
        return explanation, X_case

    # --------------------------------------------------
    def generate_chinese_explanation(self, case_dict: dict, top_n: int = 8) -> str:
        explanation, X_case = self.shap_explain(case_dict)

        sv = explanation.values[0]
        base = float(explanation.base_values[0])
        pred = base + sv.sum()

        items = sorted(
            zip(X_case.columns, sv, X_case.iloc[0]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        lines = [
            f"本模型以整體樣本平均單價 {base:.2f} 萬 / 坪為基準，",
            f"此物件預測單價約為 {pred:.2f} 萬 / 坪。",
            "",
            "主要影響因素如下：",
        ]

        for col, shap_val, data in items:
            direction = "提高" if shap_val > 0 else "降低"
            lines.append(
                f"- {self._pretty_name(col, data)}，"
                f"使單價約{direction} {abs(shap_val):.2f} 萬 / 坪"
            )

        return "\n".join(lines)

    # --------------------------------------------------
    def export_prediction_bundle(
        self,
        case_dict: dict,
        output_root: str = "output",
        top_n: int = 8,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_root, f"prediction_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        price = self.predict(case_dict)
        explanation_text = self.generate_chinese_explanation(case_dict, top_n)

        with open(os.path.join(output_dir, "prediction.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input": case_dict,
                    "predicted_price_wan_per_ping": round(price, 2),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(os.path.join(output_dir, "explanation.txt"), "w", encoding="utf-8") as f:
            f.write(explanation_text)

        explanation, X_case = self.shap_explain(case_dict)

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation[0], show=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "shap_waterfall.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        return output_dir


# ======================================================
# CLI 測試
# ======================================================
if __name__ == "__main__":

    predictor = HousePricePredictor()

    sample_case = {
        "district": "臺北市內湖區",
        "building_type": "住宅大樓",
        "main_use": "住家用",
        "building_age": 55,
        "building_area_sqm": 45,
        "floor": 8,
        "total_floors": 15,
        "main_area": 30,
        "balcony_area": 5,
        "has_parking": 1,
        "has_elevator": 1,
    }

    print(f"\n預測單價：約 {predictor.predict(sample_case):.2f} 萬 / 坪\n")
    print(predictor.generate_chinese_explanation(sample_case))
