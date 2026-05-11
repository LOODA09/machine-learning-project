import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# SAFE UNPICKLER - Skips missing TF objects in pkl files
# ============================================================
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Return a dummy placeholder for missing classes (e.g. TensorFlow)
            return type(name, (), {"__reduce__": lambda self: (type(self), ())})

def safe_pickle_load(filepath):
    with open(filepath, "rb") as f:
        return SafeUnpickler(f).load()

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Hotel Predictor Pro",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header { text-align: center; padding: 20px 0 10px 0; }
    .main-header h1 {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .main-header p { color: #888; font-size: 1.1rem; }
    .crossing-lines-bg { position: relative; overflow: hidden; padding: 20px 0; border-radius: 12px; }
    .crossing-lines-bg::before, .crossing-lines-bg::after {
        content: ''; position: absolute; top: 50%; left: 50%; width: 300%; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
        animation: crossLine1 8s linear infinite; pointer-events: none;
    }
    .crossing-lines-bg::before { transform: translate(-50%, -50%) rotate(-30deg); }
    .crossing-lines-bg::after { transform: translate(-50%, -50%) rotate(30deg); animation: crossLine2 8s linear infinite; background: linear-gradient(90deg, transparent, rgba(118,75,162,0.3), transparent); }
    @keyframes crossLine1 { 0% { transform: translate(-50%, -50%) rotate(-30deg) translateX(-50%); } 100% { transform: translate(-50%, -50%) rotate(-30deg) translateX(50%); } }
    @keyframes crossLine2 { 0% { transform: translate(-50%, -50%) rotate(30deg) translateX(50%); } 100% { transform: translate(-50%, -50%) rotate(30deg) translateX(-50%); } }
    .model-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; padding: 20px; margin: 10px 0; position: relative; overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease; backdrop-filter: blur(10px);
    }
    .model-card:hover { transform: translateY(-5px); box-shadow: 0 8px 30px rgba(102,126,234,0.3); }
    .model-card::before {
        content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
        background: conic-gradient(transparent, rgba(102,126,234,0.2), transparent 30%);
        animation: cardSpin 5s linear infinite; pointer-events: none;
    }
    @keyframes cardSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .card-content { position: relative; z-index: 1; }
    .feature-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.05), rgba(118,75,162,0.05));
        border: 1px solid rgba(102,126,234,0.2); border-radius: 12px; padding: 15px; margin-bottom: 15px;
        position: relative; overflow: hidden;
    }
    .feature-card::after { content: ''; position: absolute; top: 0; right: 0; width: 60px; height: 60px; background: linear-gradient(135deg, rgba(102,126,234,0.1), transparent); border-radius: 0 12px 0 60px; }
    .feature-label { font-weight: 600; color: #667eea; font-size: 0.95rem; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
    .result-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px; padding: 30px; text-align: center; color: white; margin: 20px 0;
        position: relative; overflow: hidden; box-shadow: 0 10px 30px rgba(17,153,142,0.3);
    }
    .result-box.canceled { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); box-shadow: 0 10px 30px rgba(203,45,62,0.3); }
    .result-icon { font-size: 3.5rem; position: relative; z-index: 1; }
    .result-text { font-size: 1.8rem; font-weight: 800; margin-top: 10px; position: relative; z-index: 1; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    # Load models and filter out dummy TF placeholders
    raw_models = safe_pickle_load("model.pkl")
    sk_models = {k: v for k, v in raw_models.items() if hasattr(v, 'predict') and hasattr(v, 'fit')}
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model_config.pkl", "rb") as f:
        config = pickle.load(f)
    with open("training_metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return sk_models, scaler, config, metrics

@st.cache_resource
def load_shap_data():
    with open("shap_data.pkl", "rb") as f:
        return pickle.load(f)

try:
    loaded_models, loaded_scaler, model_config, all_metrics = load_artifacts()
    feature_names = model_config["feature_names"]
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🏨 Navigation")
page = st.sidebar.radio("Select Analysis Module", [
    "🏠 Prediction Studio", "📊 Model Comparison", "🔍 SHAP Explainer", "👥 Guest Segments"
])

SKLEARN_MODEL_MAP = [
    ("Random Forest ⭐", "random_forest"), ("XGBoost", "xgboost"),
    ("Logistic Reg", "logistic_regression"), ("KNN", "knn"),
    ("SVM", "svm"), ("Decision Tree", "decision_tree"),
]

def predict_sklearn(input_df, models):
    results = []
    for display_name, key in SKLEARN_MODEL_MAP:
        if key not in models: continue
        model = models[key]
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else 0.5
            results.append({"Model": display_name, "Prob": float(prob), "Outcome": "Not Canceled" if pred == 1 else "Canceled"})
        except:
            results.append({"Model": display_name, "Prob": 0.0, "Outcome": "Error"})
    return results

# ============================================================
# PAGE 1: PREDICTION STUDIO
# ============================================================
if page == "🏠 Prediction Studio":
    st.markdown("""<div class="main-header"><h1>Hotel Cancellation Predictor</h1><p>Real-time booking outcome prediction powered by advanced ML models</p></div>""", unsafe_allow_html=True)
    st.markdown("### Booking Details")
    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("<div class='feature-card'><div class='feature-label'>🅿️ Parking</div></div>", unsafe_allow_html=True)
        parking_availability = st.checkbox("Parking Space Required?")
    with col_b:
        st.markdown("<div class='feature-card'><div class='feature-label'>🆕 Guest Type</div></div>", unsafe_allow_html=True)
        new_guest_flag = st.checkbox("First Time Visitor?")
    with col_c:
        st.markdown("<div class='feature-card'><div class='feature-label'>📅 Lead Time</div></div>", unsafe_allow_html=True)
        advance_days = st.number_input("Days Before Arrival", min_value=0, max_value=500, value=30)

    booking_advance = 0 if advance_days <= 1 else (1 if advance_days <= 7 else (2 if advance_days <= 30 else (3 if advance_days <= 365 else 4)))

    col_d, col_e = st.columns(2)
    with col_d:
        st.markdown("<div class='feature-card'><div class='feature-label'>💰 Room Rate</div></div>", unsafe_allow_html=True)
        room_rate = st.slider("Average Price ($)", 1, 500, 100)
    with col_e:
        st.markdown("<div class='feature-card'><div class='feature-label'>📋 Special Requests</div></div>", unsafe_allow_html=True)
        guest_requests = st.slider("Number of Requests", 0, 5, 1)

    col_f, col_g, col_h = st.columns(3)
    with col_f:
        st.markdown("<div class='feature-card'><div class='feature-label'>🌙 Stay Duration</div></div>", unsafe_allow_html=True)
        stay_option = st.selectbox("Duration", ["Day Use (0 nights)", "Short Stay (1-3)", "Week Stay (4-7)", "Two Weeks (8-14)", "Long Stay (15+)"])
        stay_duration = {"Day Use (0 nights)": 0, "Short Stay (1-3)": 1, "Week Stay (4-7)": 2, "Two Weeks (8-14)": 3, "Long Stay (15+)": 4}[stay_option]
    with col_g:
        st.markdown("<div class='feature-card'><div class='feature-label'>📆 Arrival Day</div></div>", unsafe_allow_html=True)
        day_choice = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        arrival_day = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}[day_choice]
    with col_h:
        st.markdown("<div class='feature-card'><div class='feature-label'>📊 History</div></div>", unsafe_allow_html=True)
        cancel_history = st.slider("Previous Cancel Ratio", 0.0, 1.0, 0.0, step=0.05)

    col_i, col_j = st.columns(2)
    with col_i:
        st.markdown("<div class='feature-card'><div class='feature-label'>🍽️ Meal Plan</div></div>", unsafe_allow_html=True)
        dining_choice = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    with col_j:
        st.markdown("<div class='feature-card'><div class='feature-label'>🛏️ Room Type</div></div>", unsafe_allow_html=True)
        room_choice = st.selectbox("Room Class", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])

    col_k, col_l = st.columns(2)
    with col_k:
        st.markdown("<div class='feature-card'><div class='feature-label'>🌐 Market Segment</div></div>", unsafe_allow_html=True)
        channel_choice = st.selectbox("Segment", ["Aviation", "Complementary", "Corporate", "Offline", "Online"])
    with col_l:
        st.markdown("<div class='feature-card'><div class='feature-label'>👥 Guest Count</div></div>", unsafe_allow_html=True)
        guest_choice = st.selectbox("Total Guests", ["1", "2", "3", "4", "5", "Group"])

    encoded_vector = []
    for meal in ["Meal Plan 2", "Meal Plan 3", "Not Selected"]: encoded_vector.append(1 if dining_choice == meal else 0)
    for room in ["Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]: encoded_vector.append(1 if room_choice == room else 0)
    for segment in ["Complementary", "Corporate", "Offline", "Online"]: encoded_vector.append(1 if channel_choice == segment else 0)
    for guest_cat in ["2", "3", "4", "5", "Group"]: encoded_vector.append(1 if guest_choice == guest_cat else 0)

    feature_row = [1 if parking_availability else 0, booking_advance, room_rate, guest_requests, stay_duration, arrival_day, cancel_history, 1 if new_guest_flag else 0] + encoded_vector
    input_dataframe = pd.DataFrame([feature_row], columns=feature_names)
    input_dataframe["average_price"] = loaded_scaler.transform(input_dataframe[["average_price"]])

    st.markdown("<div class='crossing-lines-bg'>", unsafe_allow_html=True)
    if st.button("🔮 Predict Booking Status", type="primary", use_container_width=True):
        st.markdown("### Ensemble Prediction Results")
        results = predict_sklearn(input_dataframe, loaded_models)
        if not results:
            st.error("No models available.")
        else:
            valid = [r for r in results if r["Outcome"] in ("Not Canceled", "Canceled")]
            nc = sum(1 for r in valid if r["Outcome"] == "Not Canceled")
            final = "Not Canceled" if nc > len(valid)/2 else "Canceled"
            css = "result-box" if final == "Not Canceled" else "result-box canceled"
            icon = "✅" if final == "Not Canceled" else "❌"
            st.markdown(f"""<div class="{css}"><div class="result-icon">{icon}</div><div class="result-text">Predicted: {final}</div><div>Agreement: {max(nc, len(valid)-nc)} / {len(valid)} Models</div></div>""", unsafe_allow_html=True)
            df_res = pd.DataFrame(results).sort_values("Prob", ascending=True)
            fig = px.bar(df_res, x="Prob", y="Model", orientation='h', color="Outcome", color_discrete_map={"Not Canceled": "#38ef7d", "Canceled": "#ef473a"}, title="Model Confidence")
            fig.update_layout(xaxis_title="Probability", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 2: MODEL COMPARISON
# ============================================================
elif page == "📊 Model Comparison":
    st.markdown("<div class='main-header'><h1>Model Evaluation Metrics</h1></div>", unsafe_allow_html=True)
    df_metrics = pd.DataFrame(all_metrics).T.reset_index().rename(columns={"index": "Model"}).sort_values("test_accuracy", ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df_metrics, x="Model", y=["train_accuracy", "test_accuracy"], barmode="group", title="Train vs Test Accuracy", color_discrete_sequence=["#667eea", "#38ef7d"])
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(df_metrics, x="test_time_sec", y="Model", orientation='h', title="Inference Time (s)", color="test_time_sec", color_continuous_scale="Purp")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### Advanced Metrics Radar")
    cats = [c for c in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1'] if c in df_metrics.columns]
    if cats:
        fig_r = go.Figure()
        for _, row in df_metrics.head(4).iterrows():
            fig_r.add_trace(go.Scatterpolar(r=[row[c] for c in cats], theta=[c.replace("test_", "").title() for c in cats], fill='toself', name=row["Model"]))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig_r, use_container_width=True)
    styled = df_metrics.style.format(precision=4)
    if "test_accuracy" in df_metrics.columns: styled = styled.background_gradient(cmap='viridis', subset=['test_accuracy'])
    if "test_f1" in df_metrics.columns: styled = styled.background_gradient(cmap='viridis', subset=['test_f1'])
    st.dataframe(styled, use_container_width=True)

# ============================================================
# PAGE 3: SHAP EXPLAINER
# ============================================================
elif page == "🔍 SHAP Explainer":
    st.markdown("""<div class="main-header"><h1>Feature Importance (SHAP)</h1><p>Explainable AI for Random Forest Predictions</p></div>""", unsafe_allow_html=True)
    try:
        shap_data = load_shap_data()
        sv = shap_data["shap_values"]; sX = shap_data["sample_X"]
        st.info("SHAP values explain how much each feature contributed to the model's prediction.")
        if isinstance(sv, list) and len(sv) == 2: mas = np.abs(sv[1]).mean(0)
        elif isinstance(sv, np.ndarray):
            mas = np.abs(sv[:, :, 1]).mean(0) if sv.ndim == 3 else np.abs(sv).mean(0)
        else: mas = np.abs(np.array(sv)).mean(0)
        fi = pd.DataFrame({"Feature": sX.columns, "Importance": mas}).sort_values("Importance", ascending=False).head(15)
        fig = px.bar(fi, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Viridis", title="Global Feature Importance (Top 15)")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fi.reset_index(drop=True).style.format({"Importance": "%.4f"}), use_container_width=True)
    except FileNotFoundError:
        st.warning("SHAP data file (`shap_data.pkl`) not found. Run the training pipeline first.")
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================================
# PAGE 4: GUEST SEGMENTS
# ============================================================
elif page == "👥 Guest Segments":
    st.markdown("""<div class="main-header"><h1>Guest Segmentation Analysis</h1><p>K-Means Clustering Profiles</p></div>""", unsafe_allow_html=True)
    st.info("The K-Means model divided guests into distinct behavioral clusters.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="model-card"><div class="card-content"><h3>💎 Premium Segment</h3><ul><li>Higher room rate ($150+)</li><li>Longer lead times</li><li>More special requests</li><li><b>Lower Cancellation Risk</b></li></ul></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="model-card"><div class="card-content"><h3>🏃 Budget / Transient</h3><ul><li>Lower room rate (&lt;$100)</li><li>Shorter lead times</li><li>Fewer requests</li><li><b>Higher Cancellation Risk</b></li></ul></div></div>""", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""<div class="model-card"><div class="card-content"><h3>🏢 Corporate</h3><ul><li>Corporate channels</li><li>Moderate rates</li><li>Weekday arrivals</li><li><b>Very Low Cancellation</b></li></ul></div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="model-card"><div class="card-content"><h3>✈️ Aviation / Special</h3><ul><li>Aviation channel</li><li>Day-use stays</li><li>Minimal requests</li><li><b>Low Cancellation</b></li></ul></div></div>""", unsafe_allow_html=True)
