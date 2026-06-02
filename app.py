import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Movie Sentiment Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("models/mnb_model.pkl")

    vectorizer = joblib.load("models/tfidf.pkl")

    return model, vectorizer

model, vectorizer = load_artifacts()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🎬 AI Sentiment Lab")

st.sidebar.markdown("---")

st.sidebar.success("Model: Multinomial Naive Bayes")

st.sidebar.info("""
Project Features

✅ NLP

✅ TF-IDF

✅ MNB

✅ Real-Time Prediction

✅ Confidence Score

✅ Interactive Dashboard
""")

# ==========================================================
# MOVIES
# ==========================================================

movies = {

    "🛩️ Top Gun: Maverick":
    "Top Gun Maverick was an amazing action movie with incredible flight sequences.",

    "🌊 Avatar: The Way of Water":
    "Avatar The Way of Water delivered stunning visuals and emotional storytelling.",

    "🕷️ Spider-Man: Across the Spider-Verse":
    "Spider Verse featured breathtaking animation and a compelling multiverse story.",

    "👠 Barbie":
    "Barbie was entertaining, funny, colorful and surprisingly meaningful.",

    "⚛️ Oppenheimer":
    "Oppenheimer was a masterpiece with outstanding performances and direction.",

    "🚀 Dune: Part Two":
    "Dune Part Two had epic visuals, intense action and world-class cinematography.",

    "😊 Inside Out 2":
    "Inside Out 2 was emotional, heartwarming and relatable for all ages.",

    "⚔️ Gladiator II":
    "Gladiator II delivered powerful action and historical drama.",

    "🦸 Deadpool & Wolverine":
    "Deadpool and Wolverine was hilarious, action-packed and entertaining.",

    "🧙 Wicked":
    "Wicked featured excellent music, performances and production design.",

    "🌺 Moana 2":
    "Moana 2 continued the adventure with memorable songs and characters.",

    "🦖 Jurassic World Rebirth":
    "Jurassic World Rebirth brought thrilling dinosaur action and adventure.",

    "🎮 A Minecraft Movie":
    "Minecraft Movie was creative, fun and appealed to gaming fans.",

    "🐰 Zootopia 2":
    "Zootopia 2 expanded its world with humor and meaningful themes.",

    "🔥 Avatar: Fire and Ash":
    "Avatar Fire and Ash offered spectacular visuals and immersive storytelling."
}

st.sidebar.markdown("---")

selected_movie = st.sidebar.selectbox(
    "🎥 Popular Movies",
    list(movies.keys())
)

if st.sidebar.button("Load Movie Review"):

    st.session_state.review_text = movies[selected_movie]

# ==========================================================
# HEADER
# ==========================================================

st.title("🎬 AI Movie Sentiment Intelligence System")

st.caption(
    "Advanced NLP Sentiment Analysis using TF-IDF + Multinomial Naive Bayes"
)

st.markdown("---")

# ==========================================================
# QUICK ACCESS MOVIES
# ==========================================================

st.subheader("🎥 Quick Movie Reviews")

col1,col2,col3,col4,col5 = st.columns(5)

with col1:
    if st.button("👠 Barbie"):
        st.session_state.review_text = movies["👠 Barbie"]

with col2:
    if st.button("⚛️ Oppenheimer"):
        st.session_state.review_text = movies["⚛️ Oppenheimer"]

with col3:
    if st.button("🚀 Dune 2"):
        st.session_state.review_text = movies["🚀 Dune: Part Two"]

with col4:
    if st.button("😊 Inside Out 2"):
        st.session_state.review_text = movies["😊 Inside Out 2"]

with col5:
    if st.button("🦸 Deadpool"):
        st.session_state.review_text = movies["🦸 Deadpool & Wolverine"]

# ==========================================================
# TEXT INPUT
# ==========================================================

default_review = st.session_state.get(
    "review_text",
    ""
)

review = st.text_area(
    "✍️ Enter Movie Review",
    value=default_review,
    height=250
)

# ==========================================================
# PREDICTION
# ==========================================================

if st.button("🚀 Analyze Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        vector = vectorizer.transform([review])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]

        confidence = np.max(probabilities) * 100

        st.markdown("---")

        if prediction == 1:

            st.success(
                f"😊 Positive Review | Confidence: {confidence:.2f}%"
            )

        else:

            st.error(
                f"😞 Negative Review | Confidence: {confidence:.2f}%"
            )

        # ==========================================
        # PROBABILITY DATAFRAME
        # ==========================================

        result_df = pd.DataFrame({

            "Sentiment": ["Negative","Positive"],

            "Probability": probabilities

        })

        st.subheader("📊 Prediction Probabilities")

        fig = px.bar(

            result_df,

            x="Sentiment",

            y="Probability",

            text="Probability"

        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================
        # GAUGE CHART
        # ==========================================

        st.subheader("🎯 Confidence Meter")

        gauge = go.Figure(go.Indicator(

            mode="gauge+number",

            value=confidence,

            title={"text":"Confidence Score"},

            gauge={
                "axis":{"range":[0,100]}
            }

        ))

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # ==========================================
        # DETAILS
        # ==========================================

        st.subheader("📋 Detailed Results")

        st.dataframe(
            result_df,
            use_container_width=True
        )

# ==========================================================
# FOOTER DASHBOARD
# ==========================================================

st.markdown("---")

col1,col2,col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "MultinomialNB"
    )

with col2:
    st.metric(
        "Vectorizer",
        "TF-IDF"
    )

with col3:
    st.metric(
        "Year",
        datetime.now().year
    )

st.markdown("---")

st.info("""
Built using:

• Python

• Streamlit

• Scikit-Learn

• TF-IDF

• Naive Bayes

• Plotly
""")