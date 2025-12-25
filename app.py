import streamlit as st
from predict import HousePricePredictor

# =========================
# 頁面設定
# =========================
st.set_page_config(
    page_title="房價估價系統",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 房價估價與 SHAP 解釋系統")
st.caption("XGBoost + 可解釋 AI（SHAP）｜依據 114 年 Q1~Q3 不動產成交資料")

# =========================
# 行政區對照表
# =========================
CITY_TOWN_MAP = {
    "臺北市": ["士林區", "大同區", "大安區", "中山區", "中正區", "內湖區", "文山區",
             "北投區", "松山區", "信義區", "南港區", "萬華區"],
    "新北市": ["板橋區", "三重區", "中和區", "永和區", "新莊區", "新店區",
             "淡水區", "汐止區", "土城區", "蘆洲區", "樹林區"],
    "桃園市": ["桃園區", "中壢區", "龜山區", "八德區", "平鎮區", "蘆竹區"],
    "臺中市": ["西屯區", "北屯區", "南屯區", "西區", "北區", "太平區"],
    "高雄市": ["三民區", "左營區", "鼓山區", "鳳山區", "前鎮區"],
}

# =========================
# 載入模型（快取）
# =========================
@st.cache_resource
def load_predictor():
    return HousePricePredictor()

predictor = load_predictor()

# =========================
# 側邊欄輸入
# =========================
st.sidebar.header("📋 房屋基本資料")

city = st.sidebar.selectbox("縣市", list(CITY_TOWN_MAP.keys()))
town = st.sidebar.selectbox("鄉鎮市區", CITY_TOWN_MAP[city])
district = f"{city}{town}"

st.sidebar.caption(f"📍 行政區：{district}")

building_type = st.sidebar.selectbox(
    "建物型態", ["住宅大樓", "華廈", "公寓", "透天厝"]
)

main_use = st.sidebar.selectbox(
    "主要用途", ["住家用", "商業用", "住商用"]
)

building_age = st.sidebar.number_input("屋齡（年）", 0, 80, 20)
main_area = st.sidebar.number_input("主建物面積（坪）", 5.0, 100.0, 30.0)
balcony_area = st.sidebar.number_input("陽台面積（坪）", 0.0, 20.0, 5.0)
floor = st.sidebar.number_input("所在樓層", 1, 100, 5)
total_floors = st.sidebar.number_input("總樓層數", 1, 100, 10)

has_parking = st.sidebar.radio("是否有車位", ["有", "無"])
has_elevator = st.sidebar.radio("是否有電梯", ["有", "無"])

# =========================
# 組合輸入資料
# =========================
case_dict = {
    "district": district,
    "building_type": building_type,
    "main_use": main_use,
    "building_age": building_age,
    "main_area": main_area,
    "balcony_area": balcony_area,
    "floor": floor,
    "total_floors": total_floors,
    "has_parking": 1 if has_parking == "有" else 0,
    "has_elevator": 1 if has_elevator == "有" else 0,
}

# =========================
# 主畫面
# =========================
st.subheader("📊 預測結果")

if "result" not in st.session_state:
    st.session_state.result = None

if st.button("🚀 開始估價"):
    with st.spinner("模型預測中，請稍候..."):
        st.session_state.result = predictor.predict(case_dict)

# =========================
# 顯示結果
# =========================
if st.session_state.result is not None:
    result = st.session_state.result

    st.success(
        f"💰 預測單價：約 **{result['predicted_price']:.1f} 萬 / 坪**"
    )

    # =========================
    # SHAP Bar 圖 + 解說
    # =========================
    st.markdown("## 🔍 影響房價最大的因素（重點分析）")

    st.info(
        "本圖顯示對本案房價影響程度最大的前五項因素。\n\n"
        "📌 長條越長，代表該因素對價格的影響越大，"
        "不論是拉高或拉低，皆表示模型在預測時高度重視該條件。"
    )

    st.pyplot(result["shap_bar_fig"], use_container_width=True)

    # =========================
    # SHAP Waterfall + 解說
    # =========================
    with st.expander("📊 查看完整價格形成過程（SHAP Waterfall 解說）"):
        st.markdown(
            """
**此圖用來說明本案房價是如何一步一步計算出來的：**

- 🔹 **左側基準值（Base value）**：模型在一般條件下的平均房價  
- 🔴 **紅色區塊**：將價格往上推高的正向條件  
- 🔵 **藍色區塊**：使價格下修的負向條件  
- 🔹 **最右側**：本案最終預測單價  

📌 每一段長度代表該條件對價格的實際影響幅度，  
所有因素加總後，即形成目前顯示的估價結果。
            """
        )

        st.pyplot(result["shap_waterfall_fig"], use_container_width=True)

    # =========================
    # 中文估價說明
    # =========================
    st.markdown("## 📝 中文估價說明")

    st.markdown(
        result["explanation"]
        .replace("•", "👉")
    )

else:
    st.info("👈 請先填寫左側資料，並點擊「開始估價」")
