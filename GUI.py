import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# ---------------------- 1. 基础配置 ----------------------
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 2. 自定义CSS：统一样式 & 蓝色按钮 ----------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
    font-family: "Helvetica Neue", Arial, sans-serif;
}
.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    padding: 20px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 18px;
    font-weight: bold;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-bottom: 15px;
}
.label-col {
    text-align: left !important;
    width: 220px;
    padding-right: 10px;
    font-size: 13px;
}
.input-col {
    flex: 1;
}
/* 全局文本左对齐 */
div[class*="stText"], div[class*="stNumberInput"], div[class*="stSelectbox"] {
    text-align: left !important;
}
/* 蓝色预测按钮 */
.stButton>button {
    background-color: #3498db !important;
    color: white !important;
    text-align: center !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    border: 2px solid white !important;   /* 添加白色边框 */
}
.stButton>button:hover {
    background-color: #2980b9 !important;
            
}
</style>
""", unsafe_allow_html=True)


# ---------------------- 3. 加载模型 & 定义特征范围 ----------------------
# 加载XGBoost模型
try:
    model = joblib.load('XGB_model.pkl')
    st.success("XGBoost model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found! Ensure 'xgboost_model.pkl' is in the current directory.")
    st.stop()
    
feature_ranges = {
    'C(%)': {"type": "numerical", "min": 8.100, "max": 88.300, "default": 55.380},
    'H(%)': {"type": "numerical", "min": 0.000, "max": 6.310, "default": 2.000},
    'O(%)': {"type": "numerical", "min": 0.300, "max": 62.530, "default": 14.870},
    'N(%)': {"type": "numerical", "min": 0.220, "max": 5.540, "default": 1.210},
    '(O+N)/C': {"type": "numerical", "min": 0.018, "max": 1.820, "default": 0.469},
    'O/C': {"type": "numerical", "min": 0.004, "max": 1.650, "default": 0.284},
    'H/C': {"type": "numerical", "min": 0.000, "max": 1.390, "default": 0.120},
    'Ash(%)': {"type": "numerical", "min": 2.750, "max": 90.670, "default": 38.443},
    'pH of Biochar': {"type": "numerical", "min": 5.310, "max": 12.620, "default": 9.270},
    'SSA(m²/g)': {"type": "numerical", "min": 0.738, "max": 553.709, "default": 15.750},
    'Initial Cd concentration (mg/L)': {"type": "numerical", "min": 1.000, "max": 500.000, "default": 100.00},
    'Rotational speed(rpm)': {"type": "numerical", "min": 120.000, "max": 4000.000, "default": 150.000},
    'Volume (L)': {"type": "numerical", "min": 0.020, "max": 0.250, "default": 0.025},
    'Concentration of biochar in water(g/L)': {"type": "numerical", "min": 0.001, "max": 20.000, "default": 1.000},
    'Adsorption temperature(℃)': {"type": "numerical", "min": 25.000, "max": 28.000, "default": 25.000},
    'Adsorption time(min)': {"type": "numerical", "min": 0.000, "max": 4760.000, "default": 150.000}
    # 'Qe(mg/g)': {"type": "numerical", "min": 0.000, "max": 310.010, "default": 18.700},
    # 'Albumin (g/L)': {"type": "numerical", "min": 35.0, "max": 50.0, "default": 40.0},
    # 'Gender': {"type": "categorical", "options": [0, 1], "label": ["Male (0)", "Female (1)"]},
    # 'Diabetes Classification': {"type": "categorical", "options": [0, 1, 2], "label": ["None (0)", "Mild (1)", "Severe (2)"]},
    # 'Hypertension Classification': {"type": "categorical", "options": [0, 1, 2], "label": ["None (0)", "Grade 1 (1)", "Grade 2 (2)"]}
    # 注释的是为了演示标签编码后的选项应该怎么填
}

# ---------------------- 4. Streamlit 界面布局 ----------------------
# 指标模块（3列布局）
with st.container():
    st.markdown('<div class="card"><h3 class="section-title">Model Input Features</h3>', unsafe_allow_html=True) # <-- 修改标题
    cols = st.columns(3)
    feature_values = []
    feature_names = list(feature_ranges.keys())

    for idx, (feature, props) in enumerate(feature_ranges.items()):
        with cols[idx % 3]:
            st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 15px;"><div class="label-col">{feature}</div><div class="input-col">', unsafe_allow_html=True)
            if props["type"] == "numerical":
                # 【修改】简化了step和format逻辑，使其更通用
                # 确定一个合理的步长，例如基于min/max/default
                default_val = float(props["default"])
                step = 0.001 if default_val < 1 else (0.1 if default_val < 10 else 1.0)
                
                value = st.number_input(
                    feature,
                    min_value=float(props["min"]),
                    max_value=float(props["max"]),
                    value=default_val,
                    step=step,
                    format="%.3f", # 使用统一的格式
                    label_visibility="collapsed"
                )
            else: # 分类特征（如果将来添加）
                value = st.selectbox(
                    feature,
                    options=props["options"],
                    format_func=lambda x: props["label"][props["options"].index(x)],
                    label_visibility="collapsed"
                )
            feature_values.append(value)
            st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # ---------------------- 5. 预测与SHAP可视化 ----------------------
# 【修改】更改按钮文本
if st.button("Predict", type="primary", use_container_width=True, key="predict_btn"):
    input_data = pd.DataFrame([feature_values], columns=feature_names)
    
    # 【修改】模型预测 (回归)
    # 1. 移除 model.predict_proba()
    # 2. 移除 "Risk Level" / "status" / "color" / "risk_prob"
    
    # model.predict() 直接返回预测值
    pred_value = model.predict(input_data)[0]

    # 【修改】计算SHAP值 (回归)
    explainer = shap.TreeExplainer(model)
    
    # 对于回归, shap_values 是一个 (n_samples, n_features) 数组
    shap_values = explainer.shap_values(input_data)
    
    # 对于回归, expected_value 是一个单独的 float
    base_value = explainer.expected_value
    
    # (1, n_features) -> (n_features,)
    single_shap = shap_values[0] 

    # 【修改】将回归结果存入session_state
    st.session_state.pred_results = {
        "pred_value": pred_value,
        "single_shap": single_shap,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "base_value": base_value # 保存 base_value (现在是float)
    }

# 【修改】显示预测结果
if "pred_results" in st.session_state:
    res = st.session_state.pred_results
    st.markdown("### Prediction Result")
    
    # 【修改】使用 st.metric 显示回归值
    st.metric(label="Predicted Value (e.g., Qe(mg/g))", value=f"{res['pred_value']:.4f}")

    # 【修改】显示SHAP瀑布图
    st.markdown("### SHAP Waterfall Plot (Feature Contribution)")
    # 【修改】更新图例说明
    st.markdown("Blue = Decrease predicted value, Red = Increase predicted value, Length = Contribution degree (Top 10 features)")
    
    single_data = pd.DataFrame([res['feature_values']], columns=res['feature_names']).iloc[0].values
    
    # 【修改】创建 Explanation 对象
    # (这个逻辑是正确的，因为 base_value 现在是 float, single_shap 是 1D array)
    shap_exp = shap.Explanation(
        values=res['single_shap'],
        base_values=res['base_value'], 
        data=single_data,
        feature_names=res['feature_names']
    )

    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig("shap_waterfall.png", dpi=300, bbox_inches='tight')
    st.image("shap_waterfall.png", use_column_width=True)

    # 【修改】显示所有特征的SHAP贡献值
    if st.checkbox("Show all features' SHAP values", key="show_shap"):
        shap_df = pd.DataFrame({
            "Feature": res['feature_names'], # <-- 修正了标签
            "Input Value": res['feature_values'],
            "SHAP Value": res['single_shap'].round(4) # <-- 修正了标签
        })
        shap_df["Absolute Contribution"] = shap_df["SHAP Value"].abs()
        shap_df_sorted = shap_df.sort_values("Absolute Contribution", ascending=False).drop("Absolute Contribution", axis=1)
        st.dataframe(shap_df_sorted, use_container_width=True)
