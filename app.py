import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App UI
st.title("📧 Email Spam Classifier")
st.write("Paste your email content below to find out if it's spam or not.")

# Input area
email_input = st.text_area("Email Text")

# Predict button
if st.button("Classify"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter some email content.")
    else:
        email_vec = vectorizer.transform([email_input])
        prediction = model.predict(email_vec)[0]
        result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
        st.success(f"Prediction: **{result}**")



