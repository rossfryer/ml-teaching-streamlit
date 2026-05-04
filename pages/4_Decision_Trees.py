"""
Decision Trees — HSC Learning Page
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

st.set_page_config(page_title="Decision Trees", page_icon="🌳", layout="wide")

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


def generate_student_support_dataset(
    n_samples: int,
    noise_level: float,
    seed: int,
) -> pd.DataFrame:
    """
    Synthetic dataset for student support classification.

    Target:
    needs_support
    0 = no (not currently flagged)
    1 = yes (flag for support)
    """
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(86, 12, n_samples), 35, 100)
    homework = np.clip(rng.normal(72, 20, n_samples), 0, 100)
    previous_score = np.clip(rng.normal(68, 15, n_samples), 0, 100)
    revision_hours = np.clip(rng.normal(6, 4, n_samples), 0, 20)

    # A non-linear, rule-ish risk pattern (good for trees):
    # - very low attendance is risky
    # - very low homework is risky
    # - low previous scores increase risk, especially with low revision
    base = (
        1.2 * (attendance < 70).astype(float)
        + 1.0 * (homework < 55).astype(float)
        + 1.1 * ((previous_score < 55) & (revision_hours < 6)).astype(float)
        + 0.8 * (revision_hours < 2).astype(float)
    )

    # Add a smooth contribution and noise so it isn't a perfect rule set.
    smooth = 0.015 * (60 - previous_score) + 0.012 * (75 - attendance) + 0.01 * (60 - homework)
    risk_score = base + smooth + rng.normal(0, noise_level, n_samples)

    prob_support = 1 / (1 + np.exp(-risk_score))
    needs_support = rng.binomial(1, prob_support)

    return pd.DataFrame(
        {
            "attendance_percent": np.round(attendance, 1),
            "homework_completion_percent": np.round(homework, 1),
            "previous_score": np.round(previous_score, 1),
            "revision_hours": np.round(revision_hours, 1),
            "needs_support": needs_support,
        }
    )


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


st.markdown('<div class="main-title">🌳 Decision Trees</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Learn simple rules from data to classify outcomes.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    <strong>Scenario:</strong> You are building a model to flag which students may need extra learning support.
    A decision tree learns a set of <strong>if/then rules</strong> from labelled examples.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What is a decision tree?", expanded=True):
    st.markdown(
        """
        A decision tree makes predictions by asking a sequence of questions such as:

        - Is attendance below 70%?
        - Is homework completion below 55%?
        - Is the previous score below 55?

        Each question splits the data into smaller groups until the tree reaches a final prediction.

        **Key idea:** more depth means more detailed rules (but can overfit).
        """
    )

with st.sidebar:
    st.header("⚙️ Controls")

    n_samples_ctl = st.slider("Number of students", 80, 1500, 400, step=40)
    noise_level_ctl = st.slider(
        "Noise level",
        0.0,
        1.5,
        0.35,
        step=0.05,
        help="Higher noise makes the labels less predictable.",
    )
    test_size_ctl = st.slider("Test data percentage", 0.10, 0.50, 0.25, step=0.05)
    seed_ctl = st.number_input("Random seed", min_value=0, value=13, step=1)

    st.subheader("Tree settings")
    max_depth = st.slider(
        "Max depth",
        min_value=1,
        max_value=12,
        value=4,
        step=1,
        help="Deeper trees learn more complex rules but may overfit.",
    )
    min_samples_leaf = st.slider(
        "Min samples per leaf",
        min_value=1,
        max_value=50,
        value=8,
        step=1,
        help="Bigger leaves force simpler rules.",
    )
    criterion = st.selectbox("Split criterion", ["gini", "entropy"])

    with st.expander("🎚️ What do these controls do?"):
        st.markdown(
            """
            **Number of students** controls dataset size.  
            **Noise level** controls how “messy” the relationship is.  
            **Test data percentage** controls how much data is held back for evaluation.  
            **Max depth** controls how many questions the tree can ask.  
            **Min samples per leaf** prevents tiny leaves that memorise the training data.  
            **Split criterion** chooses how the tree measures “best split”.
            """
        )

df = generate_student_support_dataset(
    n_samples=int(n_samples_ctl),
    noise_level=float(noise_level_ctl),
    seed=int(seed_ctl),
)

st.markdown("## 🧭 Machine learning process")
st.markdown(
    "Inspect data → choose features and target → split data → train tree → visualise rules → evaluate → explain"
)

st.divider()

st.markdown("## Step 1: Inspect the dataset")

with st.expander("📘 Data dictionary", expanded=True):
    st.markdown(
        """
        | Column | Meaning |
        |---|---|
        | `attendance_percent` | Attendance percentage |
        | `homework_completion_percent` | Homework completion percentage |
        | `previous_score` | Previous assessment score |
        | `revision_hours` | Weekly revision hours |
        | `needs_support` | Target class: 1 = flagged for support, 0 = not flagged |
        """
    )

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Target", "needs_support")

st.dataframe(df.head(20), use_container_width=True)
st.download_button(
    "⬇️ Download dataset",
    df.to_csv(index=False),
    "decision_tree_student_support.csv",
    "text/csv",
)

with st.expander("Summary statistics"):
    st.dataframe(df.describe(), use_container_width=True)

with st.expander("Class balance", expanded=True):
    balance = (
        df["needs_support"]
        .value_counts()
        .rename(index={0: "not flagged", 1: "flagged"})
        .to_frame("count")
    )
    st.dataframe(balance, use_container_width=True)
    st.markdown(
        """
        If one class is much more common, accuracy can be misleading.
        Always look at **precision**, **recall**, and the **confusion matrix** as well.
        """
    )

st.divider()

st.markdown("## Step 2: Choose the features and target")
st.markdown(
    """
    This model uses **four features** to predict the target `needs_support`.
    Decision trees work well when the relationship is **rule-like** (e.g. “low attendance AND low score”).
    """
)

feature_cols = [
    "attendance_percent",
    "homework_completion_percent",
    "previous_score",
    "revision_hours",
]

X = df[feature_cols]
y = df["needs_support"]

st.markdown("### Quick feature view")
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True)
for ax, col in zip(axes, feature_cols, strict=True):
    ax.scatter(df[col], df["needs_support"], alpha=0.35)
    ax.set_title(col.replace("_", " "))
    ax.set_xlabel(col)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.2)
axes[0].set_ylabel("needs_support")
st.pyplot(fig)
plt.close(fig)

st.divider()

st.markdown("## Step 3: Split the data")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=float(test_size_ctl),
    random_state=int(seed_ctl),
    stratify=y,
)

s1, s2, s3 = st.columns(3)
s1.metric("Total rows", len(df))
s2.metric("Training rows", len(X_train))
s3.metric("Testing rows", len(X_test))

st.markdown(
    """
    <div class="info-box">
    The training data is used to learn the tree’s rules. The testing data stays unseen until evaluation.
    Stratified splitting keeps the class mix similar in both splits.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown("## Step 4: Train the decision tree")

model = DecisionTreeClassifier(
    max_depth=int(max_depth),
    min_samples_leaf=int(min_samples_leaf),
    criterion=str(criterion),
    random_state=int(seed_ctl),
)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_m = classification_metrics(y_train, train_pred)
test_m = classification_metrics(y_test, test_pred)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test accuracy", f"{test_m['accuracy']:.3f}")
m2.metric("Test precision", f"{test_m['precision']:.3f}")
m3.metric("Test recall", f"{test_m['recall']:.3f}")
m4.metric("Test F1", f"{test_m['f1']:.3f}")

overfit_gap = train_m["accuracy"] - test_m["accuracy"]
st.markdown(
    f"""
    <div class="success-box">
    <strong>Overfitting check:</strong> training accuracy = {train_m['accuracy']:.3f}, testing accuracy = {test_m['accuracy']:.3f}
    (gap = {overfit_gap:+.3f}). A big gap often means the tree is too complex.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What does “max depth” change?"):
    st.markdown(
        """
        - A deeper tree can ask more questions, which can improve training accuracy.
        - But it may learn patterns that only exist in the training sample (overfitting).
        - Keeping `min_samples_leaf` larger usually makes the rules more reliable.
        """
    )

st.divider()

st.markdown("## Step 5: Visualise the tree (the learned rules)")

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(
    model,
    feature_names=feature_cols,
    class_names=["not flagged", "flagged"],
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    ax=ax,
)
ax.set_title("Decision tree classifier")
st.pyplot(fig)
plt.close(fig)

with st.expander("How to read the tree (detailed)", expanded=True):
    st.markdown(
        """
        Start at the top (the **root node**) and follow the arrows until you reach a **leaf**.

        **1) The split question**
        - Each node is a yes/no question like `attendance_percent <= 72.5`.
        - The number (e.g. `72.5`) is the **split threshold** the tree chose.
        - **Left branch** means the condition is **true** (≤ threshold).
        - **Right branch** means the condition is **false** (> threshold).

        **2) What the tree shows inside a node**
        - **gini** / **entropy**: how mixed the classes are at this node.
          - closer to **0** means the node is “purer” (mostly one class)
          - higher means a mix of classes
        - **samples**: how many training examples reached this node.
        - **value**: how many training examples of each class reached this node (counts per class).
        - **class**: the class the tree would predict at this node (the majority class).

        **3) Colours**
        - A darker colour usually means the node is more confident / more pure.
        - The colour also indicates which class is currently the majority.

        **4) How to interpret a path (a “rule”)**
        If you follow one path from the root to a leaf, you can read it as an if/then rule, for example:
        - IF attendance is low AND homework is low THEN predict “flagged”.
        """
    )

with st.expander("Follow one example path (what does the model do?)", expanded=False):
    st.markdown(
        """
        Pick a student in **Step 7 → Prediction explorer**, then look back at the tree and try to follow the
        same yes/no decisions:

        - Start at the root question.
        - Check whether the student’s feature value is ≤ the threshold.
        - Go left if true, right if false.
        - Repeat until you hit a leaf: that leaf’s class is the prediction.

        This is why decision trees are often called **interpretable** models: you can trace the decision.
        """
    )

st.divider()

st.markdown("## Step 6: Which features matter most?")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
st.dataframe(importances.to_frame("importance").round(3), use_container_width=True)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(importances.index, importances.values)
ax.set_title("Feature importance (higher = used more for splits)")
ax.set_ylabel("importance")
ax.set_ylim(0, max(0.01, float(importances.max()) * 1.15))
ax.grid(True, axis="y", alpha=0.25)
plt.xticks(rotation=25, ha="right")
st.pyplot(fig)
plt.close(fig)

with st.expander("Important caution"):
    st.markdown(
        """
        Feature importance shows what the tree used most to split this dataset.
        It does **not** prove real-world causation. In real systems, you would also check data quality,
        bias, and whether features are appropriate and ethical to use.
        """
    )

st.divider()

st.markdown("## Step 7: Evaluate the classifier")

cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

cm_col1, cm_col2 = st.columns([1, 1])
with cm_col1:
    confusion_display = pd.DataFrame(
        {
            "Predicted Not Flagged": [tn, fn],
            "Predicted Flagged": [fp, tp],
        },
        index=["Actual Not Flagged", "Actual Flagged"],
    )
    st.dataframe(confusion_display, use_container_width=True)

with cm_col2:
    st.markdown(
        f"""
        <div class="info-box">
        <strong>How to read this matrix</strong><br><br>
        <strong>True positives:</strong> {tp}<br>
        Flagged students correctly identified.<br><br>
        <strong>False negatives:</strong> {fn}<br>
        Flagged students missed by the model.<br><br>
        <strong>False positives:</strong> {fp}<br>
        Students incorrectly flagged.<br><br>
        <strong>True negatives:</strong> {tn}<br>
        Students correctly not flagged.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="warning-box">
    <strong>In a student support context:</strong><br>
    False negatives can be serious (missing a student who needs help). False positives can also matter (unnecessary concern or workload).
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Prediction explorer (try one student)", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        att = st.slider("attendance_percent", 35.0, 100.0, 82.0, step=0.5)
        hw = st.slider("homework_completion_percent", 0.0, 100.0, 70.0, step=0.5)
    with col_b:
        prev = st.slider("previous_score", 0.0, 100.0, 65.0, step=0.5)
        rev = st.slider("revision_hours", 0.0, 20.0, 6.0, step=0.5)

    one = pd.DataFrame(
        {
            "attendance_percent": [att],
            "homework_completion_percent": [hw],
            "previous_score": [prev],
            "revision_hours": [rev],
        }
    )
    proba = float(model.predict_proba(one)[0, 1])
    pred = int(model.predict(one)[0])
    p1, p2 = st.columns(2)
    p1.metric("Predicted probability (flagged)", f"{proba:.3f}")
    p2.metric("Predicted class", "Flagged" if pred == 1 else "Not flagged")

st.divider()

st.markdown("## ⚠️ Limitations of decision trees")
st.markdown(
    """
    <div class="warning-box">
    <strong>Important limitations:</strong><br><br>
    - Deep trees can <strong>overfit</strong> (memorise the training sample).<br>
    - Small changes in data can change the learned tree (instability).<br>
    - A single tree may not capture complex patterns as well as ensembles (e.g. random forests).<br>
    - “Feature importance” does not equal real-world causation.<br>
    - In education contexts, ethical issues (privacy, bias, fairness, transparency) matter a lot.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown("## 📝 HSC Explainer")

accuracy = test_m["accuracy"]
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
        | Decision tree | A model that uses if/then splits to make predictions |
        | Node | A point in the tree where a question splits the data |
        | Leaf | A final decision (prediction) |
        | Overfitting | When a model fits training data well but performs poorly on new data |
        | Confusion matrix | Table of correct/incorrect classifications |
        | Precision | Of predicted positives, how many were correct |
        | Recall | Of actual positives, how many were found |
        | Feature importance | A measure of how often/strongly a feature is used in splits |
        """
    )

with tabs[1]:
    st.markdown(
        f"""
        This model uses four features (attendance, homework completion, previous score, revision hours)
        to predict `needs_support` (flagged vs not flagged).

        The decision tree learned a set of rules, limited to **max depth = {int(max_depth)}**
        and **min samples per leaf = {int(min_samples_leaf)}**.

        On the test set, it achieved:
        - accuracy: `{test_m['accuracy']:.3f}`
        - precision: `{test_m['precision']:.3f}`
        - recall: `{test_m['recall']:.3f}`
        - F1: `{test_m['f1']:.3f}`

        In a student-support system, recall may be especially important because missing a student who needs support
        can have serious consequences.
        """
    )

with tabs[2]:
    answer = f"""
This system uses supervised learning because it is trained on labelled examples, where each student record includes input features and a known target class (needs support / does not need support).

A decision tree is suitable because it can learn rule-based decisions by splitting the data using if/then questions (nodes) until it reaches a final prediction (leaf). The model was trained using features such as attendance, homework completion, previous score and revision hours.

The dataset was split into training and testing data. The training data was used to learn the tree’s rules, and the testing data was used to evaluate how well the model generalises to unseen examples.

The model achieved an accuracy of {test_m['accuracy']:.3f}. However, accuracy alone is not sufficient. Precision was {test_m['precision']:.3f}, which shows how reliable flagged predictions are. Recall was {test_m['recall']:.3f}, which shows how many students who truly need support were identified. The confusion matrix also shows false positives and false negatives, which matter because missing a student who needs support can be serious.

Overall, this is a {strength} model. However, it has limitations such as overfitting if the tree is too deep, instability (small data changes can change the tree), and ethical issues such as bias and privacy. Therefore, the model should support teacher judgement rather than replace it.
""".strip()

    st.markdown(answer)
    st.download_button(
        label="⬇️ Download HSC-style answer",
        data=answer,
        file_name="decision_tree_hsc_answer.txt",
        mime="text/plain",
    )

with tabs[3]:
    st.markdown(
        """
        | Mistake | Why it is a problem | Better approach |
        |---|---|---|
        | Only reporting accuracy | Accuracy can hide important error types | Include precision, recall and a confusion matrix |
        | Making the tree very deep | Can overfit and perform poorly on new data | Use depth/leaf limits and test performance |
        | Treating feature importance as “cause” | Correlation in data is not proof | Discuss data quality and limitations |
        | Ignoring ethics | Education predictions can affect students | Mention privacy, bias, fairness, oversight |
        """
    )

st.divider()

st.markdown("## 🧠 Student reflection")
st.markdown(
    """
    Answer these questions in your notes:

    1. What happens to test accuracy when you increase max depth?
    2. When does the model start to overfit (large train–test gap)?
    3. Which errors are more serious here: false positives or false negatives? Why?
    4. Which feature did the tree use most, and what might explain that?
    5. What ethical issues could arise if this model were used in a real school?
    """
)

