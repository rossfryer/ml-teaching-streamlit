"""
Linear Regression Learning Tool (HSC Ready)

Run with:
streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Linear Regression Learning Tool",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------

st.markdown(
    """
    <style>
    .main-title {font-size: 2.4rem; font-weight: 800;}
    .subtitle {color: #555;}
    .box {background:#f5f7fb; padding:1rem; border-left:6px solid #2f80ed; border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------

st.markdown('<div class="main-title">📈 Linear Regression Learning Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explore how a model learns from data</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("⚙️ Controls")

    n_samples = st.slider("Number of houses", 50, 1000, 200, step=50)
    noise = st.slider("Noise level", 10000, 300000, 80000, step=10000)
    test_size = st.slider("Test %", 0.1, 0.5, 0.25, step=0.05)
    seed = st.number_input("Random seed", value=42)

    feature = st.selectbox(
        "Feature",
        ["floor_area", "bedrooms", "age", "distance"]
    )

    st.divider()

    with st.expander("🎚️ What do these controls do?"):
        st.markdown(
            """
            Number of houses  
            Controls how much data the model learns from  
            More data usually improves accuracy  

            Noise level  
            Adds randomness to prices  
            Higher noise makes prediction harder  

            Test percentage  
            Controls how much data is used to test the model  
            Helps evaluate performance on unseen data  

            Feature  
            The input variable used to predict price  
            Some features are stronger predictors than others  

            Random seed  
            Generates a different dataset each time  
            """
        )

# -----------------------------
# Dataset
# -----------------------------

def generate_data(n, noise, seed):
    rng = np.random.default_rng(seed)

    floor_area = rng.normal(180, 50, n)
    bedrooms = np.clip(np.round(floor_area / 50), 1, 6)
    age = rng.integers(0, 80, n)
    distance = rng.normal(15, 5, n)

    price = (
        200000
        + floor_area * 4000
        + bedrooms * 30000
        - age * 2000
        - distance * 10000
        + rng.normal(0, noise, n)
    )

    return pd.DataFrame({
        "floor_area": floor_area,
        "bedrooms": bedrooms,
        "age": age,
        "distance": distance,
        "price": price
    })

df = generate_data(n_samples, noise, int(seed))

# -----------------------------
# Step 1: Inspect
# -----------------------------

st.header("Step 1: Inspect the data")

st.dataframe(df.head())

st.write("Summary statistics")
st.dataframe(df.describe())

# -----------------------------
# Step 2: Visualise
# -----------------------------

st.header("Step 2: Explore relationship")

fig, ax = plt.subplots()
ax.scatter(df[feature], df["price"])
ax.set_xlabel(feature)
ax.set_ylabel("price")
st.pyplot(fig)

# -----------------------------
# Step 3: Split
# -----------------------------

st.header("Step 3: Split data")

X = df[[feature]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(seed)
)

st.write(f"Training: {len(X_train)} rows")
st.write(f"Testing: {len(X_test)} rows")

# -----------------------------
# Step 4: Train
# -----------------------------

st.header("Step 4: Train model")

model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_

st.metric("Slope", f"{slope:.2f}")
st.metric("Intercept", f"{intercept:.2f}")

# -----------------------------
# Step 5: Graph
# -----------------------------

st.header("Step 5: Model graph")

fig, ax = plt.subplots()

ax.scatter(X_train, y_train, label="Train")
ax.scatter(X_test, y_test, label="Test")

x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
y_line = model.predict(pd.DataFrame({feature: x_line}))

ax.plot(x_line, y_line, color="black", label="Model")

ax.legend()
st.pyplot(fig)

# -----------------------------
# Step 6: Predict
# -----------------------------

st.header("Step 6: Make a prediction")

new_x = st.slider("Choose input value", float(df[feature].min()), float(df[feature].max()))
prediction = model.predict(pd.DataFrame({feature: [new_x]}))[0]

st.metric("Predicted price", f"${prediction:,.0f}")

# -----------------------------
# Step 7: Evaluate
# -----------------------------

st.header("Step 7: Evaluate model")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{mae:,.0f}")
c2.metric("MSE", f"{mse:,.0f}")
c3.metric("RMSE", f"{rmse:,.0f}")
c4.metric("R²", f"{r2:.3f}")

# -----------------------------
# HSC Explainer
# -----------------------------

st.header("📝 HSC Explainer")

if st.button("Generate HSC-style explanation"):

    st.write(f"""
    This system uses supervised learning because it is trained on labelled data.

    The model uses {feature} to predict price using linear regression.

    The equation is:

    price = {slope:.2f} × {feature} + {intercept:.0f}

    The model achieved an R² of {r2:.3f}, meaning it explains {r2*100:.1f}% of the variation in price.

    The MAE is {mae:,.0f}, showing the average prediction error.

    This is a {"strong" if r2>0.8 else "moderate" if r2>0.5 else "weak"} model.

    However, the model is limited because it only uses one feature and does not account for all real-world factors.
    """)

# -----------------------------
# Reflection
# -----------------------------

st.header("🧠 Reflection")

st.write("""
1. Which feature worked best?
2. What happened when noise increased?
3. Why do we use test data?
4. What does R² tell us?
""")