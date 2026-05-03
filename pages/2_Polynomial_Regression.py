
"""
Polynomial Regression — HSC Learning Page
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Polynomial Regression", page_icon="〰️", layout="wide")

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

def generate_study_dataset(n_samples: int, noise_level: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    study_hours = np.clip(rng.normal(10, 5, n_samples), 0, 25)
    sleep_hours = np.clip(rng.normal(7.2, 1.2, n_samples), 3, 10)
    attendance = np.clip(rng.normal(85, 12, n_samples), 40, 100)
    practice_questions = np.clip(study_hours * rng.normal(8, 2, n_samples), 0, 250)

    score = (
        35
        + 7.5 * study_hours
        - 0.22 * (study_hours ** 2)
        + 1.2 * sleep_hours
        + 0.12 * attendance
        + rng.normal(0, noise_level, n_samples)
    )
    score = np.clip(score, 0, 100)

    return pd.DataFrame(
        {
            "study_hours": np.round(study_hours, 1),
            "sleep_hours": np.round(sleep_hours, 1),
            "attendance_percent": np.round(attendance, 1),
            "practice_questions": np.round(practice_questions, 0).astype(int),
            "exam_score": np.round(score, 1),
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

st.markdown('<div class="main-title">〰️ Polynomial Regression</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Use a curved model when a straight line is too simple.</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="info-box">
    <strong>Scenario:</strong> You are modelling student exam scores from study hours.
    At first, more study usually helps, but after a point the improvement may slow down.
    This creates a curved relationship, which polynomial regression can model.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Controls")
    n_samples = st.slider("Number of students", 50, 1000, 250, step=50)
    noise_level = st.slider("Noise level", 0.0, 25.0, 8.0, step=1.0)
    degree = st.slider("Polynomial degree", 1, 8, 2, step=1)
    test_size = st.slider("Test data percentage", 0.10, 0.50, 0.25, step=0.05)
    seed = st.number_input("Random seed", min_value=0, value=7, step=1)

    with st.expander("🎚️ What do these controls do?"):
        st.markdown(
            """
            **Number of students** controls dataset size.  
            **Noise level** controls how unpredictable scores are.  
            **Polynomial degree** controls how flexible the curve is.  
            **Test percentage** controls how much data is held back.  
            **Random seed** creates a different repeatable dataset.
            """
        )

df = generate_study_dataset(n_samples, noise_level, int(seed))

st.markdown("## 🧭 Machine learning process")
st.markdown("Inspect data → identify non-linear pattern → choose degree → split data → train model → evaluate → explain")

st.divider()

st.markdown("## Step 1: Inspect the dataset")

with st.expander("📘 Data dictionary", expanded=True):
    st.markdown(
        """
        | Column | Meaning |
        |---|---|
        | study_hours | Weekly study hours |
        | sleep_hours | Average sleep per night |
        | attendance_percent | Attendance percentage |
        | practice_questions | Approximate number of practice questions completed |
        | exam_score | Target value being predicted |
        """
    )

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Target", "exam_score")

st.dataframe(df.head(20), use_container_width=True)
st.download_button("⬇️ Download dataset", df.to_csv(index=False), "polynomial_regression_study_scores.csv", "text/csv")

with st.expander("Summary statistics"):
    st.dataframe(df.describe(), use_container_width=True)

st.divider()

st.markdown("## Step 2: Explore the relationship")

st.markdown(
    """
    Polynomial regression is useful when the pattern is curved rather than straight.
    Here, the feature is `study_hours` and the target is `exam_score`.
    """
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["study_hours"], df["exam_score"], alpha=0.65)
ax.set_title("Study hours compared with exam score")
ax.set_xlabel("study_hours")
ax.set_ylabel("exam_score")
ax.grid(True, alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.divider()

X = df[["study_hours"]]
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(seed))

model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    LinearRegression(),
)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_metrics = metrics(y_train, train_pred)
test_metrics = metrics(y_test, test_pred)

st.markdown("## Step 3: Split the data")
s1, s2, s3 = st.columns(3)
s1.metric("Total rows", len(df))
s2.metric("Training rows", len(X_train))
s3.metric("Testing rows", len(X_test))

st.markdown(
    """
    <div class="info-box">
    The model learns the curve from training data, then testing data checks whether it generalises.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown("## Step 4: Train the polynomial model")

m1, m2, m3 = st.columns(3)
m1.metric("Polynomial degree", degree)
m2.metric("Train R²", f"{train_metrics['R2']:.3f}")
m3.metric("Test R²", f"{test_metrics['R2']:.3f}")

if degree <= 1:
    st.warning("Degree 1 is equivalent to simple linear regression.")
elif degree <= 3:
    st.success("This is usually a reasonable level of flexibility for a simple teaching model.")
else:
    st.warning("Higher-degree models can overfit by following noise rather than the real pattern.")

with st.expander("What does polynomial degree mean?"):
    st.markdown(
        """
        Degree controls the complexity of the curve.

        - Degree 1 creates a straight line
        - Degree 2 creates a quadratic curve
        - Degree 3 and above can create more complex curves

        A higher degree can fit the training data better, but may perform worse on unseen test data.
        """
    )

st.divider()

st.markdown("## Step 5: Graph the model")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(X_train["study_hours"], y_train, alpha=0.6, label="Training data")
ax.scatter(X_test["study_hours"], y_test, alpha=0.85, marker="x", label="Testing data")

x_line = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 300)
y_line = model.predict(pd.DataFrame({"study_hours": x_line}))
ax.plot(x_line, y_line, linewidth=2.5, label=f"Polynomial model degree {degree}")

ax.set_title("Polynomial Regression Model")
ax.set_xlabel("study_hours")
ax.set_ylabel("exam_score")
ax.legend()
ax.grid(True, alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.divider()

st.markdown("## Step 6: Make a prediction")

new_hours = st.slider(
    "Choose study hours",
    float(df["study_hours"].min()),
    float(df["study_hours"].max()),
    float(df["study_hours"].median()),
)

prediction = model.predict(pd.DataFrame({"study_hours": [new_hours]}))[0]
prediction = float(np.clip(prediction, 0, 100))
st.metric("Predicted exam score", f"{prediction:.1f}%")

st.divider()

st.markdown("## Step 7: Evaluate the model")

e1, e2, e3, e4 = st.columns(4)
e1.metric("MAE", f"{test_metrics['MAE']:.2f}")
e2.metric("MSE", f"{test_metrics['MSE']:.2f}")
e3.metric("RMSE", f"{test_metrics['RMSE']:.2f}")
e4.metric("R²", f"{test_metrics['R2']:.3f}")

st.markdown(
    """
    | Metric | Meaning | Better result |
    |---|---|---|
    | MAE | Average absolute score error | Lower |
    | MSE | Average squared error | Lower |
    | RMSE | Typical score error | Lower |
    | R² | Proportion of variation explained | Closer to 1 |
    """
)

if train_metrics["R2"] - test_metrics["R2"] > 0.20:
    st.warning("Possible overfitting: training performance is much better than testing performance.")
else:
    st.success("The training and testing scores are reasonably close.")

with st.expander("Prediction table"):
    pred_table = pd.DataFrame(
        {
            "study_hours": X_test["study_hours"].values,
            "Actual exam score": y_test.values,
            "Predicted exam score": np.round(test_pred, 1),
            "Error": np.round(y_test.values - test_pred, 1),
            "Absolute error": np.round(np.abs(y_test.values - test_pred), 1),
        }
    )
    st.dataframe(pred_table, use_container_width=True, hide_index=True)

st.divider()

st.markdown("## 📝 HSC Explainer")

r2 = test_metrics["R2"]
mae = test_metrics["MAE"]

tabs = st.tabs(["Key terms", "Current model explanation", "Sample HSC answer", "Common mistakes"])

with tabs[0]:
    st.markdown(
        """
        | Term | HSC meaning |
        |---|---|
        | Polynomial regression | Regression model that uses powers of a feature to fit a curve |
        | Degree | Complexity of the curve |
        | Overfitting | When a model learns noise rather than the general pattern |
        | Generalisation | How well a model performs on unseen data |
        | R² | Measure of how much variation is explained |
        """
    )

with tabs[1]:
    st.markdown(
        f"""
        This model uses `study_hours` to predict `exam_score`.

        The selected polynomial degree is `{degree}`.

        The test R² is `{r2:.3f}`, meaning the model explains approximately `{r2 * 100:.1f}%` of the variation in test scores.

        The MAE is `{mae:.2f}`, meaning predictions are wrong by about this many percentage points on average.
        """
    )

with tabs[2]:
    if r2 >= 0.85:
        strength = "strong"
    elif r2 >= 0.60:
        strength = "reasonable"
    elif r2 >= 0.30:
        strength = "limited"
    else:
        strength = "weak"

    answer = f"""
This system uses supervised learning because it learns from labelled examples. Each row represents a student record with a known exam score.

Polynomial regression has been used because the relationship between study_hours and exam_score may not be perfectly straight-line. The model uses powers of study_hours to fit a curve, allowing it to represent non-linear patterns.

The model was trained using a training dataset and evaluated using a separate testing dataset. This is important because it checks whether the model generalises to unseen examples rather than simply memorising the training data.

The selected polynomial degree was {degree}. The model achieved a test R² score of {r2:.3f}, meaning it explains approximately {r2 * 100:.1f}% of the variation in exam scores. Its MAE was {mae:.2f}, meaning predictions were wrong by about this many percentage points on average.

Overall, this is a {strength} model. However, if the polynomial degree is too high, the model may overfit by following noise in the training data. The model is also limited because real exam results depend on many factors beyond study hours, such as prior knowledge, feedback, sleep, stress and teaching quality.
"""
    st.markdown(answer)
    st.download_button("⬇️ Download answer", answer, "polynomial_regression_hsc_answer.txt")

with tabs[3]:
    st.markdown(
        """
        | Mistake | Better approach |
        |---|---|
        | Thinking higher degree is always better | Compare train and test results |
        | Ignoring overfitting | Explain when the model follows noise |
        | Treating a curve as proof of causation | Say it shows a relationship, not proof |
        | Only describing the graph | Use R² and error metrics as evidence |
        """
    )

st.divider()

st.markdown("## 🧠 Reflection")
st.markdown(
    """
    1. Which degree gave the best test R²?
    2. Did a higher degree always improve the model?
    3. What signs suggest overfitting?
    4. Why might study hours have a curved relationship with score?
    5. What other features could improve the model?
    """
)
