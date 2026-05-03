
"""
Linear Regression — HSC Learning Page
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Linear Regression", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .main-title {font-size:2.4rem;font-weight:800;margin-bottom:0.2rem;}
    .subtitle {font-size:1.15rem;color:#555;margin-bottom:1.5rem;}
    .info-box {background:#f5f7fb;padding:1rem 1.2rem;border-radius:12px;border-left:6px solid #2f80ed;margin-bottom:1rem;}
    .success-box {background:#f1fbf4;padding:1rem 1.2rem;border-radius:12px;border-left:6px solid #27ae60;margin-bottom:1rem;}
    .warning-box {background:#fff8e6;padding:1rem 1.2rem;border-radius:12px;border-left:6px solid #f2a900;margin-bottom:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

def generate_housing_dataset(n_samples: int, noise_level: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    floor_area = np.clip(rng.normal(180, 55, n_samples), 60, 420)
    bedrooms = np.clip(np.round(floor_area / 45 + rng.normal(0, 0.8, n_samples)), 1, 6).astype(int)
    bathrooms = np.clip(np.round(bedrooms / 2 + rng.normal(0, 0.5, n_samples)), 1, 4).astype(int)
    age = rng.integers(0, 80, size=n_samples)
    distance = np.clip(rng.normal(14, 7, n_samples), 1, 45)
    land_size = np.clip(floor_area * rng.normal(3.2, 0.9, n_samples), 120, 1200)

    price = (
        180_000
        + floor_area * 4_200
        + bedrooms * 25_000
        + bathrooms * 35_000
        + land_size * 180
        - age * 1_800
        - distance * 9_000
        + rng.normal(0, noise_level, n_samples)
    )
    price = np.clip(price, 180_000, None)

    return pd.DataFrame(
        {
            "floor_area_m2": np.round(floor_area, 1),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "distance_to_city_km": np.round(distance, 1),
            "land_size_m2": np.round(land_size, 1),
            "sale_price": np.round(price, 0).astype(int),
        }
    )

def metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }

st.markdown('<div class="main-title">📈 Linear Regression</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict a continuous value using a straight-line model.</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="info-box">
    <strong>Scenario:</strong> You are building a model to predict house sale price from one input feature.
    Linear regression is suitable when the target is numerical and the relationship is roughly straight-line.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Controls")
    n_samples = st.slider("Number of houses", 50, 1000, 250, step=50)
    noise_level = st.slider("Noise level", 10_000, 300_000, 80_000, step=10_000)
    test_size = st.slider("Test data percentage", 0.10, 0.50, 0.25, step=0.05)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    feature = st.selectbox(
        "Input feature",
        ["floor_area_m2", "bedrooms", "bathrooms", "age_years", "distance_to_city_km", "land_size_m2"],
    )
    with st.expander("🎚️ What do these controls do?"):
        st.markdown(
            """
            **Number of houses** controls how much data the model learns from.  
            **Noise level** controls how messy the price data is.  
            **Test data percentage** controls how much data is held back for evaluation.  
            **Input feature** controls which column is used to predict sale price.  
            **Random seed** generates a different but repeatable dataset.
            """
        )

df = generate_housing_dataset(n_samples, noise_level, int(seed))

st.markdown("## 🧭 Machine learning process")
st.markdown("Inspect data → choose feature and target → split data → train model → predict → evaluate → explain")

st.divider()

st.markdown("## Step 1: Inspect the dataset")

with st.expander("📘 Data dictionary", expanded=True):
    st.markdown(
        """
        | Column | Meaning |
        |---|---|
        | floor_area_m2 | Size of the house |
        | bedrooms | Number of bedrooms |
        | bathrooms | Number of bathrooms |
        | age_years | Age of the property |
        | distance_to_city_km | Distance from city centre |
        | land_size_m2 | Land block size |
        | sale_price | Target value being predicted |
        """
    )

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Target", "sale_price")

st.dataframe(df.head(20), use_container_width=True)
st.download_button("⬇️ Download dataset", df.to_csv(index=False), "linear_regression_housing.csv", "text/csv")

with st.expander("Summary statistics"):
    st.dataframe(df.describe(), use_container_width=True)

with st.expander("Correlation with sale price"):
    st.dataframe(df.corr(numeric_only=True)["sale_price"].sort_values(ascending=False).to_frame("Correlation"), use_container_width=True)

st.divider()

st.markdown("## Step 2: Choose the feature and target")
st.markdown(f"The model will use `{feature}` to predict `sale_price`.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df[feature], df["sale_price"], alpha=0.65)
ax.set_title(f"{feature} compared with sale_price")
ax.set_xlabel(feature)
ax.set_ylabel("sale_price")
ax.grid(True, alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.divider()

X = df[[feature]]
y = df["sale_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(seed))

model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_metrics = metrics(y_train, train_pred)
test_metrics = metrics(y_test, test_pred)

slope = model.coef_[0]
intercept = model.intercept_

st.markdown("## Step 3: Split the data")
s1, s2, s3 = st.columns(3)
s1.metric("Total rows", len(df))
s2.metric("Training rows", len(X_train))
s3.metric("Testing rows", len(X_test))

st.markdown(
    """
    <div class="info-box">
    The training data is used to fit the model. The testing data is kept unseen so we can evaluate generalisation.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown("## Step 4: Train the model")

m1, m2, m3 = st.columns(3)
m1.metric("Slope", f"{slope:,.2f}")
m2.metric("Intercept", f"${intercept:,.0f}")
m3.metric("Test R²", f"{test_metrics['R2']:.3f}")

st.markdown(
    f"""
    <div class="success-box">
    <strong>Learned equation:</strong><br><br>
    sale_price = {slope:,.2f} × {feature} + {intercept:,.0f}
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What do slope and intercept mean?"):
    st.markdown(
        f"""
        The **slope** shows how much the predicted sale price changes when `{feature}` increases by one unit.  
        The **intercept** is the predicted value when `{feature}` is zero, although this may not always be realistic.
        """
    )

st.divider()

st.markdown("## Step 5: Graph the model")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(X_train[feature], y_train, alpha=0.6, label="Training data")
ax.scatter(X_test[feature], y_test, alpha=0.85, marker="x", label="Testing data")
x_line = np.linspace(df[feature].min(), df[feature].max(), 200)
y_line = model.predict(pd.DataFrame({feature: x_line}))
ax.plot(x_line, y_line, linewidth=2.5, label="Linear regression model")
ax.set_title("Linear Regression Model")
ax.set_xlabel(feature)
ax.set_ylabel("sale_price")
ax.legend()
ax.grid(True, alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.divider()

st.markdown("## Step 6: Make a prediction")

new_value = st.slider(
    f"Choose a value for {feature}",
    float(df[feature].min()),
    float(df[feature].max()),
    float(df[feature].median()),
)

prediction = model.predict(pd.DataFrame({feature: [new_value]}))[0]
st.metric("Predicted sale price", f"${prediction:,.0f}")

st.divider()

st.markdown("## Step 7: Evaluate the model")

e1, e2, e3, e4 = st.columns(4)
e1.metric("MAE", f"${test_metrics['MAE']:,.0f}")
e2.metric("MSE", f"{test_metrics['MSE']:,.0f}")
e3.metric("RMSE", f"${test_metrics['RMSE']:,.0f}")
e4.metric("R²", f"{test_metrics['R2']:.3f}")

st.markdown(
    """
    | Metric | Meaning | Better result |
    |---|---|---|
    | MAE | Average absolute error | Lower |
    | MSE | Average squared error | Lower |
    | RMSE | Typical error size | Lower |
    | R² | Proportion of variation explained | Closer to 1 |
    """
)

with st.expander("Prediction table"):
    pred_table = pd.DataFrame(
        {
            feature: X_test[feature].values,
            "Actual sale price": y_test.values,
            "Predicted sale price": np.round(test_pred, 0).astype(int),
            "Error": np.round(y_test.values - test_pred, 0).astype(int),
            "Absolute error": np.round(np.abs(y_test.values - test_pred), 0).astype(int),
        }
    )
    st.dataframe(pred_table, use_container_width=True, hide_index=True)

st.divider()

st.markdown("## 📝 HSC Explainer")

r2 = test_metrics["R2"]
mae = test_metrics["MAE"]
rmse = test_metrics["RMSE"]

if r2 >= 0.85:
    strength = "strong"
elif r2 >= 0.60:
    strength = "reasonable"
elif r2 >= 0.30:
    strength = "limited"
else:
    strength = "weak"

tabs = st.tabs(["Key terms", "Current model explanation", "Sample HSC answer", "Common mistakes"])

with tabs[0]:
    st.markdown(
        """
        | Term | HSC meaning |
        |---|---|
        | Supervised learning | Learning from labelled examples |
        | Feature | Input variable used for prediction |
        | Target | Output value being predicted |
        | Regression | Predicting a numerical value |
        | Linear regression | Fitting a straight-line model |
        | R² | Measure of how much variation the model explains |
        """
    )

with tabs[1]:
    st.markdown(
        f"""
        This model uses `{feature}` to predict `sale_price`.

        The learned equation is:

        `sale_price = {slope:,.2f} × {feature} + {intercept:,.0f}`

        The test R² is `{r2:.3f}`, meaning the model explains approximately `{r2 * 100:.1f}%` of the variation in the test data.

        The MAE is `${mae:,.0f}`, meaning the model is wrong by about this amount on average.
        """
    )

with tabs[2]:
    answer = f"""
This system uses supervised learning because it is trained using labelled examples. Each row represents a house, with input features and a known sale price.

Linear regression has been used to model the relationship between {feature} and sale_price. The model learns a straight-line equation in the form sale_price = slope × feature + intercept. In this model, the equation is sale_price = {slope:,.2f} × {feature} + {intercept:,.0f}.

The data was split into training and testing sets. The training data was used to fit the model, while the testing data was used to evaluate performance on unseen examples. This is important because a useful model should generalise beyond the data it was trained on.

The model achieved an R² score of {r2:.3f}, meaning it explains approximately {r2 * 100:.1f}% of the variation in sale prices in the test data. Its MAE was ${mae:,.0f}, meaning predictions were wrong by about this amount on average.

Overall, this is a {strength} model. However, it is limited because it uses only one feature. Real house prices are affected by many variables, including location, market conditions, property condition and renovations.
"""
    st.markdown(answer)
    st.download_button("⬇️ Download answer", answer, "linear_regression_hsc_answer.txt")

with tabs[3]:
    st.markdown(
        """
        | Mistake | Better approach |
        |---|---|
        | Saying the model is accurate without evidence | Refer to R², MAE or RMSE |
        | Confusing training and testing data | Training fits the model; testing evaluates it |
        | Saying correlation proves causation | Say the model identifies a relationship |
        | Ignoring limitations | Discuss missing features, noise and bias |
        """
    )

st.divider()

st.markdown("## 🧠 Reflection")
st.markdown(
    """
    1. Which feature gave the strongest result?
    2. What happened when noise increased?
    3. Why is testing data important?
    4. What does R² tell us?
    5. Why might this model be too simple for real house pricing?
    """
)
