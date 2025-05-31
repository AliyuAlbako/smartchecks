import streamlit as st
from models import predict_input

st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 SmartCheckAI")
st.markdown("Check if a news headline or sentence is misinformation.")

# Text input
user_input = st.text_area("Enter a news headline or sentence in English, Hausa, Fulani or Kanuri:")

# Predict button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a valid text.")
    else:
        result = predict_input(user_input)
        st.success(f"**Classification:** {result['classification']}")
        st.write(f"**Confidence Score:** {result['confidence_score']:.2f}")

# Footer
st.markdown("---")
st.caption("AI For Peace Hackathon 2025 · Team: CyberHarmony")
