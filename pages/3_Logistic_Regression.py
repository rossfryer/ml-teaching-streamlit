"""
Logistic Regression — HSC Learning Page

Run from the main app:
streamlit run streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

st.set_page_config(
    page_title="Logistic Regression",
    page_icon="✅",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size:2.4rem;
        font-weight:800;
        margin-bottom:0.2rem;
    }

    .subtitle {
        font-size:1.15rem;
        color:#555;
        margin-bottom:1.5rem;
    }

    .info-box {
        background:#f5f7fb;
        padding:1rem 1.2rem;
        border-radius:12px;
        border-left:6px solid #2f80ed;
        margin-bottom:1rem;
    }

    .success-box {
        background:#f1fbf4;
        padding:1rem 1.2rem;
        border-radius:12px;
        border-left:6px solid #27ae60;
        margin-bottom:1rem;
    }

    .warning-box {
        background:#fff8e6;
        padding:1rem 1.2rem;
        border-radius:12px;
        border-left:6px solid #f2a900;
        margin-bottom:1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Dataset generation
# -----------------------------

def generate_student_risk_dataset(
    n_samples: int,
    noise_level: float,
    seed: int,
) -> pd.DataFrame:
    """
    Generate a synthetic student-risk classification dataset.

    Target:
    at_risk
    0 = not at risk
    1 = at risk
    """

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


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title">✅ Logistic Regression</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Predict a category using probability.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    <strong>Scenario:</strong> You are building a model to predict whether a student may be academically at risk.
    Unlike linear regression, the target is a category: <strong>at risk</strong> or <strong>not at risk</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What is logistic regression?", expanded=True):
    st.markdown(
        """
        Logistic regression is used when the outcome is a **category** (class), not a continuous number.

        It predicts the **probability** that an example belongs to class 1 (e.g. *at risk*), which is always between **0** and **1**.
        A **threshold** then converts probability into a final class prediction.

        Examples of classification problems:
        - at risk / not at risk
        - spam / not spam
        - fraudulent / not fraudulent
        """
    )

# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
    st.header("⚙️ Controls")

    n_samples = st.slider(
        "Number of students",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
    )

    noise_level = st.slider(
        "Noise level",
        min_value=0.0,
        max_value=2.5,
        value=0.8,
        step=0.1,
        help="Higher noise makes the classes harder to separate.",
    )

    test_size = st.slider(
        "Test data percentage",
        min_value=0.10,
        max_value=0.50,
        value=0.25,
        step=0.05,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=21,
        step=1,
    )

    feature = st.selectbox(
        "Input feature",
        [
            "homework_completion_percent",
            "attendance_percent",
            "previous_score",
            "revision_hours",
        ],
    )

    threshold = st.slider(
        "Classification threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    with st.expander("🎚️ What do these controls do?"):
        st.markdown(
            """
            **Number of students** controls how much data the model learns from.

            **Noise level** controls how messy the data is. Higher noise makes prediction harder.

            **Test data percentage** controls how much data is kept aside to evaluate the model.

            **Input feature** controls which column is used to predict whether the student is at risk.

            **Classification threshold** controls how the model converts probability into a final class.

            **Random seed** creates a different but repeatable dataset.
            """
        )

    st.markdown(
        f"""
        <div class="info-box">
        <strong>Threshold rule</strong><br><br>

        If probability ≥ {threshold:.2f}, predict <strong>At Risk</strong>.<br>
        If probability < {threshold:.2f}, predict <strong>Not At Risk</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Generate data
# -----------------------------

df = generate_student_risk_dataset(
    n_samples=n_samples,
    noise_level=noise_level,
    seed=int(seed),
)

# -----------------------------
# Learning pathway
# -----------------------------

st.markdown("## 🧭 Machine learning process")

st.markdown(
    """
    Inspect data → choose feature and target → split data → train classifier → predict probability → evaluate → explain
    """
)

st.divider()

# -----------------------------
# Step 1
# -----------------------------

st.markdown("## Step 1: Inspect the dataset")

st.markdown(
    """
    Before training a model, inspect the dataset carefully.

    In this example, each row represents one student. The model will use one feature to predict whether the student is at risk.
    """
)

with st.expander("📘 Data dictionary", expanded=True):
    st.markdown(
        """
        | Column | Meaning |
        |---|---|
        | `homework_completion_percent` | Percentage of homework completed |
        | `attendance_percent` | Student attendance percentage |
        | `previous_score` | Previous assessment score |
        | `revision_hours` | Weekly revision hours |
        | `at_risk` | Target class: 1 = at risk, 0 = not at risk |
        """
    )

c1, c2, c3 = st.columns(3)

c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Target", "at_risk")

st.markdown("### Dataset preview")
st.dataframe(df.head(20), use_container_width=True)

st.download_button(
    label="⬇️ Download dataset",
    data=df.to_csv(index=False),
    file_name="logistic_regression_student_risk.csv",
    mime="text/csv",
)

with st.expander("Summary statistics"):
    st.dataframe(df.describe(), use_container_width=True)

with st.expander("Class balance", expanded=True):
    balance = (
        df["at_risk"]
        .value_counts()
        .rename(index={0: "not at risk", 1: "at risk"})
        .to_frame("count")
    )

    st.dataframe(balance, use_container_width=True)

    st.markdown(
        """
        Class balance matters because a model can appear accurate if one class is much more common than the other.

        For example, if 90% of students are not at risk, a model could predict “not at risk” every time and still appear to have high accuracy.
        """
    )

st.divider()

# -----------------------------
# Step 2
# -----------------------------

st.markdown("## Step 2: Choose the feature and target")

st.markdown(
    f"""
    For this model:

    - input feature: `{feature}`
    - target: `at_risk`

    The target has two possible values:

    - `0` means not at risk
    - `1` means at risk
    """
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df[feature],
    df["at_risk"],
    alpha=0.45,
)

ax.set_title(f"{feature} compared with at_risk")
ax.set_xlabel(feature)
ax.set_ylabel("at_risk")
ax.set_yticks([0, 1])
ax.grid(True, alpha=0.25)

st.pyplot(fig)
plt.close(fig)

with st.expander("How to read this graph"):
    st.markdown(
        """
        The target is categorical, so the points appear at either 0 or 1.

        Logistic regression does not try to draw a straight line through these points.

        Instead, it learns a probability curve that estimates how likely class 1 is.
        """
    )

st.divider()

# -----------------------------
# Train model
# -----------------------------

X = df[[feature]]
y = df["at_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=int(seed),
    stratify=y,
)

model = LogisticRegression()
model.fit(X_train, y_train)

prob_train = model.predict_proba(X_train)[:, 1]
prob_test = model.predict_proba(X_test)[:, 1]

pred_train = (prob_train >= threshold).astype(int)
pred_test = (prob_test >= threshold).astype(int)

accuracy = accuracy_score(y_test, pred_test)
precision = precision_score(y_test, pred_test, zero_division=0)
recall = recall_score(y_test, pred_test, zero_division=0)
f1 = f1_score(y_test, pred_test, zero_division=0)

coef = model.coef_[0][0]
intercept = model.intercept_[0]

# -----------------------------
# Step 3
# -----------------------------

st.markdown("## Step 3: Split the data")

s1, s2, s3 = st.columns(3)

s1.metric("Total rows", len(df))
s2.metric("Training rows", len(X_train))
s3.metric("Testing rows", len(X_test))

st.markdown(
    """
    <div class="info-box">
    The model learns from the training data. The testing data is kept separate so we can check how well the model works on unseen examples.
    Stratified splitting is used so both classes are represented fairly in the training and testing data.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------
# Step 4
# -----------------------------

st.markdown("## Step 4: Train the logistic regression model")

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

        In this model, the coefficient is `{coef:.3f}`.
        """
    )

with st.expander("Why is logistic regression different from linear regression?"):
    st.markdown(
        """
        Linear regression predicts a continuous number, such as a price or score.

        Logistic regression predicts the probability of a class.

        This makes logistic regression suitable for classification tasks such as:

        - at risk or not at risk
        - spam or not spam
        - approved or rejected
        - pass or fail
        """
    )

st.divider()

# -----------------------------
# Step 5
# -----------------------------

st.markdown("## Step 5: Graph the probability curve")

with st.expander("📊 Why is the graph S-shaped?", expanded=True):
    st.markdown(
        """
        Logistic regression uses an S-shaped curve called a **sigmoid curve**.

        This is useful because probabilities must stay between 0 and 1.

        A straight line would not work well because it could predict values below 0 or above 1.

        The curve means:

        - low input values may give probability close to 0
        - high input values may give probability close to 1
        - middle values often show uncertainty
        """
    )

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.scatter(
    X_train[feature],
    y_train,
    alpha=0.35,
    label="Training data",
)

ax.scatter(
    X_test[feature],
    y_test,
    alpha=0.65,
    marker="x",
    label="Testing data",
)

x_line = np.linspace(df[feature].min(), df[feature].max(), 300)
prob_line = model.predict_proba(pd.DataFrame({feature: x_line}))[:, 1]

ax.plot(
    x_line,
    prob_line,
    linewidth=2.5,
    label="Predicted probability of at risk",
)

ax.axhline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.2f}",
)

ax.set_title("Logistic Regression Probability Curve")
ax.set_xlabel(feature)
ax.set_ylabel("Probability / class")
ax.set_yticks([0, threshold, 1])
ax.legend()
ax.grid(True, alpha=0.25)

st.pyplot(fig)
plt.close(fig)

st.divider()

# -----------------------------
# Step 6
# -----------------------------

st.markdown("## Step 6: Make a prediction")

new_value = st.slider(
    f"Choose a value for {feature}",
    min_value=float(df[feature].min()),
    max_value=float(df[feature].max()),
    value=float(df[feature].median()),
)

new_probability = model.predict_proba(
    pd.DataFrame({feature: [new_value]})
)[0, 1]

new_class = int(new_probability >= threshold)

p1, p2 = st.columns(2)

p1.metric("Predicted probability of at risk", f"{new_probability:.3f}")
p2.metric("Predicted class", "At risk" if new_class == 1 else "Not at risk")

if new_class == 1:
    st.warning("The model predicts this student is at risk.")
else:
    st.success("The model predicts this student is not at risk.")

with st.expander("🔍 Probability vs classification"):
    st.markdown(
        f"""
        Logistic regression first produces a probability.

        For this example, the predicted probability is `{new_probability:.3f}`.

        The model then compares this probability with the threshold `{threshold:.2f}`.

        - If the probability is greater than or equal to the threshold, the model predicts `At risk`
        - If it is below the threshold, the model predicts `Not at risk`

        The probability is often more informative than the final class because it shows uncertainty.
        """
    )

st.divider()

# -----------------------------
# Step 7
# -----------------------------

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

with st.expander("⚖️ Understanding precision vs recall"):
    st.markdown(
        """
        In classification, different mistakes have different consequences.

        In this education example:

        - false positive means a student is incorrectly flagged as at risk
        - false negative means an at-risk student is missed

        High precision means fewer false positives.

        High recall means fewer false negatives.

        In a student support system, recall may be especially important because missing an at-risk student could prevent them from receiving help.
        """
    )

with st.expander("🎯 Confusion matrix (advanced)", expanded=False):
    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])

    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    cm_col1, cm_col2 = st.columns([1, 1])

    with cm_col1:
        confusion_display = pd.DataFrame(
            {
                "Predicted Not At Risk": [true_negatives, false_negatives],
                "Predicted At Risk": [false_positives, true_positives],
            },
            index=["Actual Not At Risk", "Actual At Risk"],
        )

        st.dataframe(confusion_display, use_container_width=True)

    with cm_col2:
        st.markdown(
            f"""
            <div class="info-box">
            <strong>How to read this matrix</strong><br><br>

            <strong>True negatives:</strong> {true_negatives}<br>
            Students correctly predicted as not at risk.<br><br>

            <strong>True positives:</strong> {true_positives}<br>
            At-risk students correctly identified.<br><br>

            <strong>False positives:</strong> {false_positives}<br>
            Students incorrectly flagged as at risk.<br><br>

            <strong>False negatives:</strong> {false_negatives}<br>
            At-risk students missed by the model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="warning-box">
        <strong>Important:</strong><br>
        In this example, a false negative may be more serious than a false positive because it means an at-risk student was missed.
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("📉 How the threshold changes the metrics (advanced)", expanded=False):
    st.markdown(
        """
        This graph shows how performance changes as the classification threshold changes.

        - Lower thresholds usually increase recall but may reduce precision.
        - Higher thresholds usually increase precision but may reduce recall.
        """
    )

    threshold_values = np.arange(0.05, 0.96, 0.05)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for t in threshold_values:
        threshold_predictions = (prob_test >= t).astype(int)

        accuracy_scores.append(accuracy_score(y_test, threshold_predictions))
        precision_scores.append(precision_score(y_test, threshold_predictions, zero_division=0))
        recall_scores.append(recall_score(y_test, threshold_predictions, zero_division=0))
        f1_scores.append(f1_score(y_test, threshold_predictions, zero_division=0))

    threshold_df = pd.DataFrame(
        {
            "Threshold": threshold_values,
            "Accuracy": accuracy_scores,
            "Precision": precision_scores,
            "Recall": recall_scores,
            "F1 score": f1_scores,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(threshold_df["Threshold"], threshold_df["Accuracy"], label="Accuracy")
    ax.plot(threshold_df["Threshold"], threshold_df["Precision"], label="Precision")
    ax.plot(threshold_df["Threshold"], threshold_df["Recall"], label="Recall")
    ax.plot(threshold_df["Threshold"], threshold_df["F1 score"], label="F1 score")

    ax.axvline(
        threshold,
        linestyle="--",
        label=f"Current threshold = {threshold:.2f}",
    )

    ax.set_title("How changing the threshold affects classification metrics")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Metric score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    t1, t2, t3 = st.tabs(["Interpretation", "Threshold table", "Prediction table"])

    with t1:
        st.markdown(
            """
            The threshold controls how cautious or aggressive the model is when predicting the At Risk class.

            If the threshold is low:
            - the model predicts At Risk more often
            - recall usually increases
            - false positives may increase

            If the threshold is high:
            - the model predicts At Risk less often
            - precision may increase
            - false negatives may increase

            In a student support system, recall may be especially important because missing an at-risk student could have serious consequences.
            """
        )

    with t2:
        st.dataframe(
            threshold_df.round(3),
            use_container_width=True,
            hide_index=True,
        )

    with t3:
        prediction_table = pd.DataFrame(
            {
                feature: X_test[feature].values,
                "Actual class": y_test.values,
                "Predicted probability": np.round(prob_test, 3),
                "Predicted class": pred_test,
            }
        )

        st.dataframe(
            prediction_table,
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# -----------------------------
# Limitations
# -----------------------------

st.markdown("## ⚠️ Limitations of logistic regression")

st.markdown(
    """
    <div class="warning-box">
    <strong>Important limitations:</strong><br><br>

    - Logistic regression models relatively simple relationships.
    - It assumes a smooth transition between classes.
    - It is sensitive to missing, poor-quality or biased data.
    - This example only uses one feature.
    - Predictions are probabilities, not certainties.
    - Human judgement is still needed in real educational decisions.

    In real systems, a model like this should support decision-making, not replace teachers or pastoral staff.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------
# HSC Explainer
# -----------------------------

st.markdown("## 📝 HSC Explainer")

if accuracy >= 0.85:
    strength = "strong"
elif accuracy >= 0.70:
    strength = "reasonable"
elif accuracy >= 0.55:
    strength = "limited"
else:
    strength = "weak"

tabs = st.tabs(
    [
        "Key terms",
        "Current model explanation",
        "Sample HSC answer",
        "Common mistakes",
    ]
)

with tabs[0]:
    st.markdown(
        """
        | Term | HSC meaning |
        |---|---|
        | Classification | Predicting a category or class |
        | Logistic regression | A classification algorithm that predicts probability |
        | Probability | The model's estimated likelihood of a class |
        | Threshold | Probability cut-off used to choose a class |
        | Accuracy | Proportion of correct predictions |
        | Precision | How reliable positive predictions are |
        | Recall | How many actual positives were detected |
        | F1 score | Balance between precision and recall |
        | Confusion matrix | Table of correct and incorrect classifications |
        | False positive | Incorrectly predicting the positive class |
        | False negative | Incorrectly predicting the negative class |
        """
    )

with tabs[1]:
    st.markdown(
        f"""
        This model uses `{feature}` to predict whether a student is `at_risk`.

        The model predicts a probability between 0 and 1.

        If the probability is at least `{threshold:.2f}`, the model predicts `at_risk = 1`.

        The current model achieved:

        - accuracy: `{accuracy:.3f}`
        - precision: `{precision:.3f}`
        - recall: `{recall:.3f}`
        - F1 score: `{f1:.3f}`

        Accuracy shows the overall proportion of correct predictions, but precision and recall are also important because different types of mistakes have different consequences.
        """
    )

with tabs[2]:
    answer = f"""
This system uses supervised learning because it is trained using labelled data, where each example includes input features and a known target classification.

Logistic regression has been used because the problem involves predicting a categorical outcome, specifically whether a student is at risk. Unlike linear regression, logistic regression outputs a probability between 0 and 1 using a sigmoid function.

The model uses {feature} as its input feature and learns a relationship between this feature and the likelihood that a student is at risk. A threshold of {threshold:.2f} is used to convert the predicted probability into a final classification.

The dataset was split into training and testing data. The training data was used to fit the model, while the testing data was used to evaluate how well the model generalises to unseen data.

The model achieved an accuracy of {accuracy:.3f}. However, accuracy alone is not sufficient for evaluation. The precision was {precision:.3f}, meaning that when the model predicts a student is at risk, it is correct this proportion of the time. The recall was {recall:.3f}, meaning that this proportion of actual at-risk students were correctly identified. The F1 score of {f1:.3f} provides a balance between these two measures.

The confusion matrix also helps evaluate the model because it shows true positives, true negatives, false positives and false negatives. In this context, a false negative may be particularly serious because an at-risk student could be missed.

Overall, this is a {strength} model. However, it is limited because it relies on a single feature and cannot capture all factors affecting student performance. Additionally, classification errors may have real consequences, and ethical considerations such as bias, fairness and privacy must be taken into account. Therefore, the model should be used as a support tool rather than a standalone decision-making system.
"""

    st.markdown(answer)

    st.download_button(
        label="⬇️ Download HSC-style answer",
        data=answer,
        file_name="logistic_regression_hsc_answer.txt",
        mime="text/plain",
    )

with tabs[3]:
    st.markdown(
        """
        | Mistake | Why it is a problem | Better approach |
        |---|---|---|
        | Calling logistic regression a regression model only | It is mainly used for classification | Explain that it predicts probability for a class |
        | Relying only on accuracy | Accuracy can hide false positives and false negatives | Include precision, recall, F1 and confusion matrix |
        | Ignoring the threshold | The threshold changes the final classification | Explain how probability becomes a class |
        | Treating predictions as facts | The model estimates likelihood, not certainty | Use language such as “predicts” or “estimates” |
        | Ignoring ethics | Educational predictions can affect students | Discuss privacy, bias, fairness and human oversight |
        """
    )

st.divider()

# -----------------------------
# Reflection
# -----------------------------

st.markdown("## 🧠 Student reflection")

st.markdown(
    """
    Answer these questions in your notes:

    1. Which feature produced the best classification result?
    2. What happens when the threshold is lowered?
    3. What happens when the threshold is raised?
    4. Why can accuracy be misleading?
    5. Why might recall matter in an at-risk student model?
    6. What is the difference between a probability and a class prediction?
    7. What ethical issues could arise if this type of model were used in a real school?
    """
)