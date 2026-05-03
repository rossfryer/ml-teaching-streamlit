
"""
Logistic Regression — HSC Learning Page
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Logistic Regression", page_icon="✅", layout="wide")

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

def generate_student_risk_dataset(n_samples: int, noise_level: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    homework_completion = np.clip(rng.normal(72, 20, n_samples), 0, 100)
    attendance = np.clip(rng.normal(86, 12, n_samples), 40, 100)
    previous_score = np.clip(rng.normal(68, 15, n_samples), 0, 100)
    revision_hours = np.clip(rng.normal(6, 4, n_samples), 0, 20)

    risk_score = (
        -0.055 * homework_completion
        -0.045 * attendance
        -0.050 * previous_score
        -0.080 * revision_hours
        + 9.5
        + rng.normal(0, noise_level, n_samples)
    )

    probability_at_risk = 1 / (1 + np.exp(-risk_score))
    at_risk = rng.binomial(1, probability_at_risk)

    return pd.DataFrame(
        {
            "homework_completion_percent": np.round(homework_completion, 1),
            "attendance_percent": np.round(attendance, 1),
            "previous_score": np.round(previous_score, 1),
            "revision_hours": np.round(revision_hours, 1),
            "at_risk": at_risk,
        }
    )

st.markdown('<div class="main-title">✅ Logistic Regression</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict a category using probability.</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="info-box">
    <strong>Scenario:</strong> You are building a model to predict whether a student may be academically at risk.
    Unlike linear regression, the target is a category: <strong>at risk</strong> or <strong>not at risk</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Controls")
    n_samples = st.slider("Number of students", 50, 1000, 300, step=50)
    noise_level = st.slider("Noise level", 0.0, 2.5, 0.8, step=0.1)
    test_size = st.slider("Test data percentage", 0.10, 0.50, 0.25, step=0.05)
    seed = st.number_input("Random seed", min_value=0, value=21, step=1)
    feature = st.selectbox(
        "Input feature",
        ["homework_completion_percent", "attendance_percent", "previous_score", "revision_hours"],
    )
    threshold = st.slider("Classification threshold", 0.10, 0.90, 0.50, step=0.05)

    with st.expander("🎚️ What do these controls do?"):
        st.markdown(
            """
            **Number of students** controls dataset size.  
            **Noise level** makes classification messier.  
            **Test percentage** controls evaluation size.  
            **Input feature** controls what evidence the model uses.  
            **Threshold** controls the probability cut-off for predicting at risk.
            """
        )

df = generate_student_risk_dataset(n_samples, noise_level, int(seed))

st.markdown("## 🧭 Machine learning process")
st.markdown("Inspect data → choose feature and category target → split data → train classifier → predict probability → evaluate → explain")

st.divider()

st.markdown("## Step 1: Inspect the dataset")

with st.expander("📘 Data dictionary", expanded=True):
    st.markdown(
        """
        | Column | Meaning |
        |---|---|
        | homework_completion_percent | Percentage of homework completed |
        | attendance_percent | Student attendance percentage |
        | previous_score | Previous assessment score |
        | revision_hours | Weekly revision hours |
        | at_risk | Target class: 1 = at risk, 0 = not at risk |
        """
    )

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Target", "at_risk")

st.dataframe(df.head(20), use_container_width=True)
st.download_button("⬇️ Download dataset", df.to_csv(index=False), "logistic_regression_student_risk.csv", "text/csv")

with st.expander("Class balance"):
    balance = df["at_risk"].value_counts().rename(index={0: "not at risk", 1: "at risk"}).to_frame("count")
    st.dataframe(balance, use_container_width=True)
    st.markdown("Class balance matters because a model can look accurate if one class is much more common than the other.")

st.divider()

st.markdown("## Step 2: Choose the feature and target")
st.markdown(f"The model will use `{feature}` to predict whether `at_risk` is 0 or 1.")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df[feature], df["at_risk"], alpha=0.45)
ax.set_title(f"{feature} compared with at_risk")
ax.set_xlabel(feature)
ax.set_ylabel("at_risk")
ax.set_yticks([0, 1])
ax.grid(True, alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.divider()

X = df[[feature]]
y = df["at_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(seed), stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

prob_test = model.predict_proba(X_test)[:, 1]
pred_test = (prob_test >= threshold).astype(int)

accuracy = accuracy_score(y_test, pred_test)
precision = precision_score(y_test, pred_test, zero_division=0)
recall = recall_score(y_test, pred_test, zero_division=0)
f1 = f1_score(y_test, pred_test, zero_division=0)

st.markdown("## Step 3: Split the data")
s1, s2, s3 = st.columns(3)
s1.metric("Total rows", len(df))
s2.metric("Training rows", len(X_train))
s3.metric("Testing rows", len(X_test))

st.markdown(
    """
    <div class="info-box">
    Logistic regression learns from training data and is evaluated on unseen testing data.
    Stratified splitting is used so both classes are represented fairly.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown("## Step 4: Train the logistic regression model")

coef = model.coef_[0][0]
intercept = model.intercept_[0]

m1, m2, m3 = st.columns(3)
m1.metric("Coefficient", f"{coef:.3f}")
m2.metric("Intercept", f"{intercept:.3f}")
m3.metric("Accuracy", f"{accuracy:.3f}")

st.markdown(
    """
    <div class="success-box">
    Logistic regression predicts a probability between 0 and 1.
    If the probability is above the selected threshold, the model predicts class 1: at risk.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What does the coefficient mean?"):
    st.markdown(
        f"""
        The coefficient shows how `{feature}` affects the model's predicted risk.

        - A positive coefficient means higher values increase predicted risk
        - A negative coefficient means higher values reduce predicted risk
        """
    )

st.divider()

st.markdown("## Step 5: Graph the probability curve")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(X_train[feature], y_train, alpha=0.35, label="Training data")
ax.scatter(X_test[feature], y_test, alpha=0.65, marker="x", label="Testing data")

x_line = np.linspace(df[feature].min(), df[feature].max(), 300)
prob_line = model.predict_proba(pd.DataFrame({feature: x_line}))[:, 1]

ax.plot(x_line, prob_line, linewidth=2.5, label="Predicted probability of at risk")
ax.axhline(threshold, linestyle="--", label=f"Threshold = {threshold:.2f}")
ax.set_title("Logistic Regression Probability Curve")
ax.set_xlabel(feature)
ax.set_ylabel("Probability / class")
ax.set_yticks([0, threshold, 1])
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

new_probability = model.predict_proba(pd.DataFrame({feature: [new_value]}))[0, 1]
new_class = int(new_probability >= threshold)

p1, p2 = st.columns(2)
p1.metric("Predicted probability of at risk", f"{new_probability:.3f}")
p2.metric("Predicted class", "At risk" if new_class == 1 else "Not at risk")

if new_class == 1:
    st.warning("The model predicts this student is at risk.")
else:
    st.success("The model predicts this student is not at risk.")

st.divider()

st.markdown("## Step 7: Evaluate the classifier")

e1, e2, e3, e4 = st.columns(4)
e1.metric("Accuracy", f"{accuracy:.3f}")
e2.metric("Precision", f"{precision:.3f}")
e3.metric("Recall", f"{recall:.3f}")
e4.metric("F1 score", f"{f1:.3f}")

st.markdown(
    """
    | Metric | Meaning | Better result |
    |---|---|---|
    | Accuracy | Overall proportion of correct predictions | Higher |
    | Precision | Of predicted at-risk students, how many really were at risk | Higher |
    | Recall | Of actual at-risk students, how many were found | Higher |
    | F1 score | Balance between precision and recall | Higher |
    """
)

cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["Actual not at risk", "Actual at risk"],
    columns=["Predicted not at risk", "Predicted at risk"],
)

with st.expander("Confusion matrix", expanded=True):
    st.dataframe(cm_df, use_container_width=True)
    st.markdown(
        """
        A confusion matrix shows correct and incorrect predictions for each class.
        This is useful when one type of mistake is more serious than another.
        """
    )

with st.expander("Prediction table"):
    pred_table = pd.DataFrame(
        {
            feature: X_test[feature].values,
            "Actual class": y_test.values,
            "Predicted probability": np.round(prob_test, 3),
            "Predicted class": pred_test,
        }
    )
    st.dataframe(pred_table, use_container_width=True, hide_index=True)

st.divider()

st.markdown("## 📝 HSC Explainer")

if accuracy >= 0.85:
    strength = "strong"
elif accuracy >= 0.70:
    strength = "reasonable"
elif accuracy >= 0.55:
    strength = "limited"
else:
    strength = "weak"

tabs = st.tabs(["Key terms", "Current model explanation", "Sample HSC answer", "Common mistakes"])

with tabs[0]:
    st.markdown(
        """
        | Term | HSC meaning |
        |---|---|
        | Classification | Predicting a category or class |
        | Logistic regression | Classification algorithm that predicts probability |
        | Threshold | Probability cut-off used to choose a class |
        | Accuracy | Proportion of correct predictions |
        | Precision | How reliable positive predictions are |
        | Recall | How many actual positives were detected |
        | Confusion matrix | Table of correct and incorrect classifications |
        """
    )

with tabs[1]:
    st.markdown(
        f"""
        This model uses `{feature}` to predict whether a student is `at_risk`.

        The model predicts a probability. If the probability is at least `{threshold:.2f}`, the model predicts `at_risk = 1`.

        The accuracy is `{accuracy:.3f}`, precision is `{precision:.3f}`, recall is `{recall:.3f}`, and F1 score is `{f1:.3f}`.
        """
    )

with tabs[2]:
    answer = f"""
This system uses supervised learning because it is trained on labelled examples. Each row contains student information and a known target class indicating whether the student is at risk.

Logistic regression has been used because the target is categorical rather than numerical. The model predicts the probability that a student belongs to the at-risk class. If the predicted probability is greater than or equal to the threshold of {threshold:.2f}, the model classifies the student as at risk.

The model uses {feature} as its input feature. It was trained using a training dataset and evaluated on a separate testing dataset. This helps measure whether the model can generalise to unseen student records.

The model achieved an accuracy of {accuracy:.3f}, meaning it correctly classified approximately {accuracy * 100:.1f}% of the test examples. Its precision was {precision:.3f}, recall was {recall:.3f}, and F1 score was {f1:.3f}. These metrics are important because accuracy alone can be misleading, especially if one class is more common than the other.

Overall, this is a {strength} classifier. However, the model is limited because it uses only one feature. A real student support system would need more evidence, careful human judgement, privacy protection and awareness of bias before making decisions about students.
"""
    st.markdown(answer)
    st.download_button("⬇️ Download answer", answer, "logistic_regression_hsc_answer.txt")

with tabs[3]:
    st.markdown(
        """
        | Mistake | Better approach |
        |---|---|
        | Calling logistic regression a regression model only | Explain it is used for classification |
        | Relying only on accuracy | Include precision, recall and confusion matrix |
        | Ignoring threshold | Explain how probability becomes a class |
        | Ignoring ethical issues | Discuss privacy, bias and human oversight |
        | Treating predictions as facts | Say the model estimates risk |
        """
    )

st.divider()

st.markdown("## 🧠 Reflection")
st.markdown(
    """
    1. Which feature produced the best classification result?
    2. What happens when the threshold changes?
    3. Why can accuracy be misleading?
    4. Why might recall matter in an at-risk student model?
    5. What ethical issues could arise in this type of system?
    """
)
