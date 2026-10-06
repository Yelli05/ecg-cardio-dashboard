import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Where the backend lives. Locally it's your own machine; on Day 9 we point it at Render.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
N_STEPS = 187
CLASS_NAMES = ["Normal (N)", "Supraventricular ectopic (S)", "Ventricular ectopic (V)",
               "Fusion (F)", "Unknown / paced (Q)"]

st.set_page_config(page_title="ECG Cardio Dashboard", page_icon="🫀", layout="wide")
st.title("ECG Heartbeat Classifier")
st.caption("A 1D-CNN trained on the MIT-BIH Arrhythmia dataset. "
           "Educational demo, not a medical device.")


@st.cache_data
def load_samples():
    return pd.read_csv(Path(__file__).parent / "sample_heartbeats.csv")


def get_beat():
    """Return (values, recorded_label_or_None) from the sidebar choice, or (None, None)."""
    source = st.sidebar.radio("Heartbeat source", ["Sample heartbeat", "Upload CSV"])

    if source == "Sample heartbeat":
        samples = load_samples()
        options = [f"Beat {i + 1}" for i in range(len(samples))]
        choice = st.sidebar.selectbox("Choose a beat", options)
        row = samples.iloc[options.index(choice)]
        label = int(row["label"])
        st.sidebar.info(f"Recorded label: **{CLASS_NAMES[label]}**")
        return [float(v) for v in row.iloc[:N_STEPS]], label

    file = st.sidebar.file_uploader("CSV with one heartbeat (187 values in one row)", type="csv")
    if file is None:
        return None, None
    nums = pd.read_csv(file, header=None).apply(pd.to_numeric, errors="coerce")
    row = nums.dropna(how="all").iloc[0].dropna().tolist()
    if len(row) == N_STEPS + 1:          # a 188th value is the label column, ignore it
        row = row[:N_STEPS]
    if len(row) != N_STEPS:
        st.sidebar.error(f"Need {N_STEPS} values in the first row, found {len(row)}.")
        return None, None
    return [float(v) for v in row], None


values, recorded_label = get_beat()

if values is None:
    st.info("Pick a sample beat or upload a CSV in the sidebar.")
    st.stop()

st.subheader("Selected heartbeat")
st.line_chart(pd.DataFrame({"amplitude": values}))

if st.button("Classify heartbeat", type="primary"):
    try:
        with st.spinner("Contacting the model (the first request can take a while)..."):
            r = requests.post(f"{API_URL}/predict", json={"values": values}, timeout=90)
        r.raise_for_status()
        result = r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get a prediction from the backend at {API_URL}. Details: {e}")
        st.stop()

    summary = f"**{result['label']}**, confidence {result['confidence']:.1%}"
    if result["is_abnormal"]:
        st.error(f"Abnormal pattern detected: {summary}")
    else:
        st.success(f"Normal: {summary}")

    st.subheader("Class probabilities")
    st.bar_chart(pd.Series(result["probabilities"]))