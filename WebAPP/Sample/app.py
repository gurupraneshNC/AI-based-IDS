import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Set Page Config
st.set_page_config(page_title="AI Network IDS", page_icon="🛡️", layout="wide")

# Title
st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("### ⚡ Real-time Multi-Model Analysis")
st.markdown("Analyzing traffic simultaneously with **CNN Binary**, **LSTM Binary**, **CNN Multi-class**, and **LSTM Multi-class** models.")

# --- Constants & Paths ---
DATA_PATH_TRAIN = r"data/UNSW_NB15_testing-set(in).csv"
DATA_PATH_TEST = r"data/UNSW_NB15_training-set(in).csv"
MODEL_DIR = "models"

# --- 1. Data Loading & Preprocessing (Cached) ---
@st.cache_resource
def load_data_and_preprocessors():
    if not os.path.exists(DATA_PATH_TRAIN):
        st.error(f"❌ Training file not found at {DATA_PATH_TRAIN}")
        return None, None, None, None

    # Load Data
    df_train = pd.read_csv(DATA_PATH_TRAIN)
    if os.path.exists(DATA_PATH_TEST):
        df_test = pd.read_csv(DATA_PATH_TEST)
    else:
        df_test = df_train 

    drop_cols = ['id', 'label', 'attack_cat']
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    
    # --- Refit Encoders ---
    cat_cols = df_train[feature_cols].select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        unique_vals = pd.concat([df_train[col], df_test[col]]).unique()
        le.fit(unique_vals.astype(str))
        encoders[col] = le
        df_train[col] = le.transform(df_train[col].astype(str))

    # --- Refit Scaler ---
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    
    # --- Fit Target Encoder ---
    target_le = LabelEncoder()
    target_le.fit(pd.concat([df_train['attack_cat'], df_test['attack_cat'] if os.path.exists(DATA_PATH_TEST) else df_train['attack_cat']]).astype(str))
    
    return feature_cols, encoders, scaler, target_le

feature_cols, encoders, scaler, target_le = load_data_and_preprocessors()

# --- 2. Model Loading Helper ---
@st.cache_resource
def load_ids_model(model_name):
    possible_paths = [
        os.path.join(MODEL_DIR, model_name),
        model_name,
        os.path.join("gurupraneshnc", "ai-based-ids", "AI-based-IDS-45d5e3338b43a35f67f05f5f7e8e7d367f9f6066", model_name)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path)
            except Exception as e:
                st.error(f"Error loading {path}: {e}")
                return None
    return None

# --- 3. Input Section ---
st.sidebar.header("Input Data")
input_method = st.sidebar.radio("Choose Input Method", ["Random Test Sample", "Manual Input", "Upload CSV Row"])
input_df = None

if input_method == "Random Test Sample":
    if st.sidebar.button("Generate Random Sample"):
        if os.path.exists(DATA_PATH_TEST):
            df_test_raw = pd.read_csv(DATA_PATH_TEST)
            sample = df_test_raw.sample(1)
            st.write("**Selected Test Sample (Raw Data):**")
            st.dataframe(sample)
            input_df = sample.reset_index(drop=True)
        else:
            st.warning("Test data file not found.")

elif input_method == "Upload CSV Row":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.write("**Uploaded Data:**")
        st.dataframe(input_df)

elif input_method == "Manual Input":
    st.info("Configure traffic parameters in the sidebar.")
    with st.sidebar.form("manual_input"):
        st.write("Key Traffic Features")
        def get_opts(col, default): return encoders[col].classes_ if encoders else default

        dur = st.number_input("Duration", value=0.000011)
        proto = st.selectbox("Protocol", get_opts('proto', ['tcp','udp']))
        service = st.selectbox("Service", get_opts('service', ['dns','http']))
        state = st.selectbox("State", get_opts('state', ['FIN','INT']))
        spkts = st.number_input("Source Packets", value=10)
        dpkts = st.number_input("Dest Packets", value=10)
        sbytes = st.number_input("Source Bytes", value=1000)
        dbytes = st.number_input("Dest Bytes", value=500)
        rate = st.number_input("Rate", value=100000.0)
        
        if st.form_submit_button("Analyze Traffic") and feature_cols:
            input_data = {col: [0] for col in feature_cols}
            input_data.update({'dur': [dur], 'proto': [proto], 'service': [service], 'state': [state],
                               'spkts': [spkts], 'dpkts': [dpkts], 'sbytes': [sbytes], 'dbytes': [dbytes], 'rate': [rate]})
            input_df = pd.DataFrame(input_data)

# --- 4. Multi-Model Analysis Logic ---
if input_df is not None and feature_cols is not None:
    st.divider()
    st.header("🔍 Analysis Results")
    
    # --- A. Preprocessing (Common for all) ---
    try:
        processed_df = input_df.copy()
        for col in feature_cols:
            if col not in processed_df.columns:
                processed_df[col] = 0
        X_input = processed_df[feature_cols].copy()
        
        for col, le in encoders.items():
            X_input[col] = X_input[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0])

        X_scaled_base = scaler.transform(X_input)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # --- B. Define Models & Configurations ---
    # Based on our analysis of your files:
    # Binary models -> 44 features
    # Multi models  -> 43 features
    
    model_configs = [
        {"name": "CNN Binary", "file": "cnn_binary.keras", "type": "binary", "arch": "cnn", "features": 44},
        {"name": "LSTM Binary", "file": "lstm_binary.keras", "type": "binary", "arch": "lstm", "features": 44},
        {"name": "CNN Multi-class", "file": "cnnmulti.keras", "type": "multi", "arch": "cnn", "features": 43},
        {"name": "LSTM Multi-class", "file": "lstm_multi.keras", "type": "multi", "arch": "lstm", "features": 43},
    ]

    # --- C. Create Columns for Display ---
    cols = st.columns(4)

    # --- D. Loop Through Each Model ---
    for i, config in enumerate(model_configs):
        with cols[i]:
            st.subheader(config["name"])
            
            # 1. Load Model
            model = load_ids_model(config["file"])
            if model is None:
                st.error("Model not found")
                continue

            try:
                # 2. Fix Feature Count (Pad/Trim to 43 or 44)
                target_features = config["features"]
                current_features = X_scaled_base.shape[1]
                
                if current_features < target_features:
                    padding = np.zeros((X_scaled_base.shape[0], target_features - current_features))
                    X_ready = np.hstack((X_scaled_base, padding))
                elif current_features > target_features:
                    X_ready = X_scaled_base[:, :target_features]
                else:
                    X_ready = X_scaled_base

                # 3. Reshape based on Architecture
                # LSTM -> (1, Features)
                # CNN -> (Features, 1)
                if config["arch"] == "lstm":
                    X_final = X_ready.reshape(X_ready.shape[0], 1, target_features)
                else:
                    X_final = X_ready.reshape(X_ready.shape[0], target_features, 1)

                # 4. Predict
                prediction = model.predict(X_final, verbose=0)

                # 5. Display Result
                if config["type"] == "binary":
                    prob = prediction[0][0]
                    if prob > 0.5:
                        st.error("🚨 **ATTACK**")
                    else:
                        st.success("✅ **NORMAL**")
                    st.metric("Probability", f"{prob:.4f}")
                    st.progress(float(prob))
                    
                else: # Multi-class
                    class_idx = np.argmax(prediction)
                    # Safe decode
                    if class_idx < len(target_le.classes_):
                        cat = target_le.inverse_transform([class_idx])[0]
                    else:
                        cat = f"Class {class_idx}"
                    
                    st.info(f"**{cat.upper()}**")
                    st.metric("Confidence", f"{prediction[0][class_idx]:.4f}")
                    
                    # Small top 3 table
                    st.markdown("---")
                    st.caption("Top Matches:")
                    top_3 = np.argsort(prediction[0])[-3:][::-1]
                    for idx in top_3:
                        c_name = target_le.inverse_transform([idx])[0] if idx < len(target_le.classes_) else str(idx)
                        st.caption(f"{c_name}: {prediction[0][idx]:.0%}")

            except Exception as e:
                st.error(f"Error: {e}")