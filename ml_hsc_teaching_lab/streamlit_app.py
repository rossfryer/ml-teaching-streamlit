
"""
HSC Machine Learning Teaching Lab — Home Page

Run:
streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="HSC ML Teaching Lab",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #f5f7fb;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #2f80ed;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #f1fbf4;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #27ae60;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff8e6;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #f2a900;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #e6e8ef;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧠 HSC Machine Learning Teaching Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A guided, interactive lab for learning regression, classification, model evaluation and HSC-style explanation.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    <strong>Purpose of this tool</strong><br><br>
    This app helps students understand how machine learning models are built, tested and evaluated.
    Each page uses a realistic scenario, a generated dataset, visualisations, metrics and HSC-style explanation support.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🧭 Learning pathway")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="card">
        <h3>📈 Linear Regression</h3>
        <p>Predict a continuous value using a straight-line model.</p>
        <p><strong>Example:</strong> predicting house price from floor area.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
        <h3>〰️ Polynomial Regression</h3>
        <p>Predict a continuous value using a curved model.</p>
        <p><strong>Example:</strong> modelling non-linear study hours and test score patterns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="card">
        <h3>✅ Logistic Regression</h3>
        <p>Predict a category using probability.</p>
        <p><strong>Example:</strong> predicting whether a student is at risk or not at risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("## 📊 What problem does supervised learning solve?")

st.markdown(
    """
Supervised learning is used when you have examples where the correct answer is already known.

For example, a dataset might contain:

- input features such as house size, age or distance from the city
- a target value such as sale price
- many rows of past examples

The model learns a relationship between the inputs and the target so that it can make predictions on new data.
"""
)

st.markdown(
    """
    <div class="success-box">
    <strong>Core idea:</strong><br>
    Supervised learning uses labelled examples to learn a mapping from inputs to outputs.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🧪 How students should use the lab")

st.markdown(
    """
1. Open a page from the sidebar  
2. Read the scenario  
3. Inspect the generated dataset  
4. Change one control at a time  
5. Train and visualise the model  
6. Interpret the evaluation metrics  
7. Use the HSC explainer to practise exam-style writing  
"""
)

st.markdown(
    """
    <div class="warning-box">
    <strong>Experiment tip:</strong><br>
    Change one variable at a time. For example, increase noise while keeping sample size the same,
    then observe what happens to R², error and the graph.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 📝 HSC Software Engineering connection")

st.markdown(
    """
This tool supports understanding of:

- supervised learning
- regression and classification
- training and testing data
- feature selection
- data quality
- model accuracy
- evaluation metrics
- limitations and ethical considerations
- explaining and evaluating machine learning systems in extended responses
"""
)

st.markdown("## 🚀 Start")

st.markdown(
    """
    <div class="success-box">
    Use the sidebar to open <strong>1 Linear Regression</strong>, then work through the pages in order.
    </div>
    """,
    unsafe_allow_html=True,
)
