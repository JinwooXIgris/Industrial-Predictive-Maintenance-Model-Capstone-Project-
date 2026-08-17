"""
streamlit_app.py
Streamlit web application for real-time predictive maintenance diagnostics,
input parameter validation, telemetry anomaly visualization, and historical data analytics.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust system path to import modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    import config
    from inference import PredictiveMaintenanceInference
except ModuleNotFoundError:
    # If run outside directory
    sys.path.append(os.path.join(BASE_DIR, "Capstone_Model_Deployment"))
    import config
    from inference import PredictiveMaintenanceInference

# Page layout configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Custom premium styling
st.markdown("""
    <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .main-header {
            font-size: 40px;
            font-weight: 800;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 20px;
            color: #4B5563;
            text-align: center;
            margin-bottom: 40px;
        }
        .status-box {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .status-safe {
            background-color: #D1FAE5;
            color: #065F46;
            border: 2px solid #34D399;
        }
        .status-fail {
            background-color: #FEE2E2;
            color: #991B1B;
            border: 2px solid #F87171;
        }
    </style>
""", unsafe_allow_html=True)


# Load historical data cached to prevent UI lag
@st.cache_data
def load_historical_stats(path: str):
    """
    Loads raw predictive maintenance dataset to display historical statistics.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Compute key constants for anomaly diagnostics (Z-score calculators)
        stats_dict = {
            "air_temp_mean": df["Air temperature [K]"].mean(),
            "air_temp_std": df["Air temperature [K]"].std(),
            "proc_temp_mean": df["Process temperature [K]"].mean(),
            "proc_temp_std": df["Process temperature [K]"].std(),
            "rot_speed_mean": df["Rotational speed [rpm]"].mean(),
            "rot_speed_std": df["Rotational speed [rpm]"].std(),
            "torque_mean": df["Torque [Nm]"].mean(),
            "torque_std": df["Torque [Nm]"].std(),
            "tool_wear_mean": df["Tool wear [min]"].mean(),
            "tool_wear_std": df["Tool wear [min]"].std(),
            "total_records": len(df),
            "failure_rate": df["Target"].mean() * 100,
            "type_counts": df["Type"].value_counts().to_dict(),
            "df_sample": df.sample(1000, random_state=42) # Sample for distribution plots
        }
        return stats_dict, df
    else:
        # Fallback dictionary if file path fails
        stats_dict = {
            "air_temp_mean": 300.0, "air_temp_std": 2.0,
            "proc_temp_mean": 310.0, "proc_temp_std": 1.0,
            "rot_speed_mean": 1500.0, "rot_speed_std": 180.0,
            "torque_mean": 40.0, "torque_std": 10.0,
            "tool_wear_mean": 100.0, "tool_wear_std": 64.0,
            "total_records": 10000,
            "failure_rate": 3.39,
            "type_counts": {"L": 6000, "M": 3000, "H": 1000},
            "df_sample": None
        }
        return stats_dict, None


# Load stats
stats_data, raw_df = load_historical_stats(config.DATA_PATH)

# Main Title UI
st.markdown("<div class='main-header'>⚙️ Enterprise Predictive Maintenance Diagnostics</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Real-time machine telemetry classification and structural anomaly analysis</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# SIDEBAR CONTROLS & MODEL SELECTION
# -------------------------------------------------------------------------
st.sidebar.header("🔧 Telemetry Configurations")

# Sidebar slider inputs for machine telemetry features
air_temp = st.sidebar.slider("Air Temperature (Kelvin)", 280.0, 315.0, 298.1, step=0.1)
proc_temp = st.sidebar.slider("Process Temperature (Kelvin)", 290.0, 325.0, 308.6, step=0.1)
rot_speed = st.sidebar.slider("Rotational Speed (RPM)", 500, 3500, 1550, step=10)
torque = st.sidebar.slider("Torque (Nm)", 0.0, 100.0, 42.8, step=0.1)
tool_wear = st.sidebar.slider("Tool Wear Accumulation (min)", 0, 260, 50, step=1)
machine_type = st.sidebar.selectbox("Machine variant class", ["L", "M", "H"], index=1)

# Organize telemetries
input_telemetry = {
    "air_temp": air_temp,
    "proc_temp": proc_temp,
    "rot_speed": float(rot_speed),
    "torque": torque,
    "tool_wear": float(tool_wear),
    "machine_type": machine_type
}

st.sidebar.markdown("---")
st.sidebar.header("🎯 Model Parameters")

# Check for model file existences to populate available models
available_models = []
for model_key in ["logistic_regression", "decision_tree", "random_forest", "hist_gradient_boosting", "xgboost"]:
    model_file = os.path.join(config.MODELS_DIR, f"{model_key}_final_model.pkl")
    if os.path.exists(model_file):
        available_models.append(model_key)

if not available_models:
    st.sidebar.error("⚠️ No trained models found in `/models`. Please run the training pipeline first!")
    st.error("Missing model artifacts! Please run `capstone_main.ipynb` to train models and generate pickle files.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Select Classification Model", available_models)

# Load inference engine
inference_engine = PredictiveMaintenanceInference(model_name=selected_model_name)

# Allow manual override of threshold
decision_thresh = st.sidebar.slider(
    "Decision Probability Threshold",
    0.05, 0.95,
    float(inference_engine.threshold),
    step=0.01,
    help="Telemetry is classified as FAILURE if predicted probability exceeds this threshold."
)

# -------------------------------------------------------------------------
# DIAGNOSTICS & PREDICTIONS
# -------------------------------------------------------------------------
col_main, col_stats = st.columns([3, 2])

with col_main:
    st.subheader("📊 Live Telemetry Diagnostics")
    
    # Run prediction
    try:
        results = inference_engine.predict(input_telemetry, custom_threshold=decision_thresh)
        
        # Display safety status
        if results["prediction_class"] == 0:
            st.markdown(
                f"<div class='status-box status-safe'>✅ STATUS: NORMAL OPERATION ({results['confidence_pct']:.2f}% Confidence)</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='status-box status-fail'>🚨 STATUS: MACHINE FAILURE RISK ({results['confidence_pct']:.2f}% Confidence)</div>",
                unsafe_allow_html=True
            )
            
        # Display warnings if inputs are physically implausible or out of training bounds
        if results["warnings"]:
            for warn in results["warnings"]:
                st.warning(f"⚠️ {warn}")
                
        # Metric Cards for Live Inputs
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Failure Probability", f"{results['probability'] * 100:.2f}%", help="Calculated risk probability")
        m_col2.metric("Operational Risk Level", results["risk_level"])
        m_col3.metric("Decision Threshold", f"{results['decision_threshold']:.2f}")
        
    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("🕵️ Telemetry Anomaly Analyzer (Z-Scores)")
    st.write("Below we analyze how far current telemetry is deviating from normal training set averages (expressed in Standard Deviations). Values outside [-2, 2] indicate structural anomalies.")
    
    # Calculate Z-Scores
    z_scores = {
        "Air Temp": (air_temp - stats_data["air_temp_mean"]) / stats_data["air_temp_std"],
        "Process Temp": (proc_temp - stats_data["proc_temp_mean"]) / stats_data["proc_temp_std"],
        "Rotational Speed": (rot_speed - stats_data["rot_speed_mean"]) / stats_data["rot_speed_std"],
        "Torque": (torque - stats_data["torque_mean"]) / stats_data["torque_std"],
        "Tool Wear": (tool_wear - stats_data["tool_wear_mean"]) / stats_data["tool_wear_std"]
    }
    
    z_df = pd.DataFrame(list(z_scores.items()), columns=["Sensor", "Z-Score"])
    z_df["Status"] = z_df["Z-Score"].apply(lambda x: "Normal" if abs(x) <= 2.0 else ("Warning (High)" if x > 2.0 else "Warning (Low)"))
    
    # Plot Z-scores
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = z_df["Z-Score"].apply(lambda x: "tomato" if abs(x) > 2.0 else "skyblue")
    
    sns.barplot(x="Z-Score", y="Sensor", data=z_df, palette=colors.tolist(), ax=ax, edgecolor="black")
    ax.axvline(x=2.0, color="red", linestyle="--", alpha=0.7, label="Anomalous Bounds")
    ax.axvline(x=-2.0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlim([-4.5, 4.5])
    ax.set_xlabel("Standard Deviations (Z-Score)")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# -------------------------------------------------------------------------
# HISTORICAL DATA ANALYTICS & MODEL EXPLANATIONS
# -------------------------------------------------------------------------
with col_stats:
    st.subheader("📈 Historical Machine Analytics")
    st.write(f"Based on historical telemetry database of **{stats_data['total_records']}** machines.")
    
    # 1. Total Failure rate metric
    st.info(f"⚙️ **Historical Database Failure Rate:** {stats_data['failure_rate']:.2f}% (Rare Event)")
    
    # 2. Plot failure rate by machine quality type
    if raw_df is not None:
        st.write("**Failure Rates by Machine Quality Variant:**")
        # Compute rates
        type_fail = raw_df.groupby("Type")["Target"].mean() * 100
        type_fail = type_fail.reindex(["L", "M", "H"])
        
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.barplot(x=type_fail.index, y=type_fail.values, palette="Blues_r", edgecolor="black", ax=ax2)
        ax2.set_ylabel("Failure Rate (%)")
        ax2.set_xlabel("Machine Variant (L=Low, M=Medium, H=High)")
        for i, val in enumerate(type_fail.values):
            ax2.text(i, val + 0.1, f"{val:.2f}%", ha="center", weight="bold")
        ax2.set_ylim([0, max(type_fail.values) + 1.0])
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
        # 3. Common Failure modes breakdown
        st.write("**Failure Distribution by Type:**")
        fail_modes = raw_df[raw_df["Failure Type"] != "No Failure"]["Failure Type"].value_counts()
        
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        sns.barplot(y=fail_modes.index, x=fail_modes.values, palette="Reds_r", edgecolor="black", ax=ax3)
        ax3.set_xlabel("Occurrences Count")
        ax3.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    else:
        st.warning("Historical CSV database not found. Showing cached statistical proxies.")
        
# Footer metadata details
st.markdown("---")
st.write("ℹ️ **Model Architecture Information:**")
st.write(f"- Active Model variant: `{selected_model_name.upper()}`")
st.write("- Target Metric: F1-Score (optimized via Stratified 5-Fold Grid Search)")
st.write("- Features processed: [scaled telemetry sensors, scaled physical interaction proxies, one-hot encoded variant categories]")
