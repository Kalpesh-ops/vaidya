# app.py — HuggingFace Space entry point

import streamlit as st
import requests

API_URL = "http://localhost:8080"

st.set_page_config(
    page_title="Vaidya — Classical Ayurvedic Knowledge",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Vaidya")
st.subheader("Citation-Grounded Ayurvedic Knowledge from Caraka-Saṃhitā")

st.info(
    "Ask questions in English, Hindi, or Sanskrit. "
    "Every answer is sourced directly from the Caraka-Saṃhitā with verse citations."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of sources", 1, 10, 5)
    sthana = st.selectbox(
        "Filter by Sthana (optional)",
        ["All", "Sutrasthana", "Nidanasthana", "Vimanasthana",
         "Sharirasthana", "Cikitsasthana", "Kalpasthana", "Siddhisthana"]
    )
    st.markdown("---")
    st.caption("Sources: Caraka-Saṃhitā (Priyavrata Śarmā ed., Chaukhambha Orientalia)")
    st.caption("Model: Qwen2.5-72B on AMD MI300X")

query = st.text_input(
    "Enter your query:",
    placeholder="e.g., What are the properties of Vata dosha? / वात दोष के गुण क्या हैं?"
)

if st.button("Search", type="primary") and query:
    with st.spinner("Searching classical texts..."):
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "sthana_filter": None if sthana == "All" else sthana
            }
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
            data = resp.json()

            if data["confident"]:
                st.success("✓ Sources found in Caraka-Saṃhitā")
            else:
                st.warning("⚠ No direct source found above confidence threshold")

            st.markdown("### Response")
            st.write(data["response"])

            if data["citations"]:
                st.markdown("### Citations")
                for c in data["citations"]:
                    st.code(c)

            st.markdown("---")
            st.warning(data["disclaimer"])

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption(
    "Vaidya is an AI research system. All information is for educational purposes only. "
    "Consult a certified Ayurvedic practitioner for medical advice."
)