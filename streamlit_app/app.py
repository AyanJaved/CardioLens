import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

API_BASE = os.environ.get("CARDIOLENS_API_URL")
if not API_BASE:
    try:
        API_BASE = st.secrets["CARDIOLENS_API_URL"]
    except Exception:
        API_BASE = "http://localhost:8000"
API_URL = API_BASE + "/predict"

st.set_page_config(page_title="CardioLens", page_icon="🫀")
st.title("CardioLens")
st.caption("Cardiomegaly detection from chest X-rays")

st.warning(
    "Educational/portfolio project only. Not a diagnostic tool. "
    "Not validated for clinical use."
)

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded X-ray", width=300)

    if st.button("Analyze"):
        with st.spinner("Running inference..."):
            uploaded.seek(0)
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            response = requests.post(API_URL, files=files)

        if response.status_code != 200:
            st.error(f"API error: {response.status_code}")
        else:
            result = response.json()
            st.subheader(result["label"])
            st.metric("Probability", f"{result['probability']:.3f}")
            st.caption(f"Decision threshold: {result['threshold']:.3f}")

            heatmap_bytes = base64.b64decode(result["heatmap_base64"])
            heatmap = Image.open(io.BytesIO(heatmap_bytes))
            st.image(heatmap, caption="Grad-CAM heatmap", width=300)