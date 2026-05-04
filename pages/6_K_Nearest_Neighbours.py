"""
K-Nearest Neighbours — HSC Learning Page
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

st.set_page_config(page_title="K-Nearest Neighbours", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
    .main-title {font-size:2.4rem;font-weight:800;margin-bottom:0.2rem;}
    .subtitle {font-size:1.15rem;color:#555;margin-bottom:1.5rem;}
    .info-box {background:#f5f7fb;padding:1rem 1.2rem;border-radius:12px;border-left:6px solid #2f80ed;margin-bottom:1rem;}
    .success-box {background:#f1fbf4;padding:1rem 1.2rem;border-radius:12px;border-left:6px solid #27ae60;margin-bottom:1rem;}
    .warning-box {background:#fff8e6;padding:1rem 1.2rem;border-radius:12px;border-left:6px solid #f2a900;margin-bottom:1rem;}
    code {background: rgba(0,0,0,0.04); padding: 0.1rem 0.25rem; border-radius: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def generate_two_moons_dataset(n_samples: int, noise: float, seed: int) -> pd.DataFrame:
    """
    Create a 2D "two moons" dataset without extra dependencies.
    Target:
      class_label: 0 or 1
    """
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    t0 = rng.uniform(0, np.pi, size=n0)
    x0 = np.cos(t0)
    y0 = np.sin(t0)

    t1 = rng.uniform(0, np.pi, size=n1)
    x1 = 1.0 - np.cos(t1)
    y1 = 0.2 - np.sin(t1)

    X_data = np.vstack([np.c_[x0, y0], np.c_[x1, y1]])
    y_data = np.array([0] * n0 + [1] * n1)

    X_data = X_data + rng.normal(0, noise, size=X_data.shape)

    return pd.DataFrame(
        {
            "feature_1": np.round(X_data[:, 0], 3),
            "feature_2": np.round(X_data[:, 1], 3),
            "class_label": y_data.astype(int),
        }
    )


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_decision_boundary(model: KNeighborsClassifier, X_df: pd.DataFrame, y_ser: pd.Series, title: str) -> None:
    x_min, x_max = float(X_df["feature_1"].min()) - 0.5, float(X_df["feature_1"].max()) + 0.5
    y_min, y_max = float(X_df["feature_2"].min()) - 0.5, float(X_df["feature_2"].max()) + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 260),
        np.linspace(y_min, y_max, 260),
    )
    grid = pd.DataFrame({"feature_1": xx.ravel(), "feature_2": yy.ravel()})

    # Probability is easier to interpret than just class labels.
    proba = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    contour = ax.contourf(xx, yy, proba, levels=30, cmap="RdYlBu", alpha=0.85)
    fig.colorbar(contour, ax=ax, label="P(class 1)")

    ax.scatter(
        X_df["feature_1"],
        X_df["feature_2"],
        c=y_ser,
        cmap="bwr",
        edgecolor="k",
        s=28,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("feature_1")
    ax.set_ylabel("feature_2")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


st.markdown('<div class="main-title">🧭 K-Nearest Neighbours (KNN)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Classify an example by looking at the most similar examples.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    <strong>Scenario:</strong> You want to classify a new student into one of two groups based on two measurements.
    KNN compares the new student to the <strong>most similar</strong> students in the dataset and uses a vote.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What is KNN?", expanded=True):
    st.markdown(
        """
        K-nearest neighbours (KNN) is a simple supervised learning algorithm:

        - It stores the training data.
        - To classify a new point, it finds the **k closest points** (nearest neighbours).
        - The predicted class is the **majority vote** of those neighbours (or a **distance-weighted** vote).

        **Key idea:** small k can overfit (wiggly boundary), large k can underfit (too smooth).
        """
    )

with st.sidebar:
    st.header("⚙️ Controls")

    n_samples_ctl = st.slider("Number of points", 100, 2500, 650, step=50)
    noise_ctl = st.slider("Noise level", 0.00, 0.40, 0.10, step=0.01)
    test_size_ctl = st.slider("Test data percentage", 0.10, 0.50, 0.25, step=0.05)
    seed_ctl = st.number_input("Random seed", min_value=0, value=23, step=1)

    st.subheader("KNN settings")
    k = st.slider("k (number of neighbours)", 1, 41, 9, step=2)
    weights = st.selectbox("Vote weighting", ["uniform", "distance"])
    p = st.selectbox(
        "Distance type",
        options=[1, 2],
        format_func=lambda v: "Manhattan (p=1)" if v == 1 else "Euclidean (p=2)",
        help="p=2 is the usual straight-line distance. p=1 uses grid-like distance.",
    )

    with st.expander("🎚️ What do these controls do?"):
        st.markdown(
            """
            **k** controls how many neighbours vote.  
            **uniform** means each neighbour votes equally.  
            **distance** means closer neighbours count more.  
            **Distance type (p)** changes what “close” means.
            """
        )

df = generate_two_moons_dataset(
    n_samples=int(n_samples_ctl),
    noise=float(noise_ctl),
    seed=int(seed_ctl),
)

st.markdown("## 🧭 Machine learning process")
st.markdown("Inspect data → choose features and target → split data → train KNN → visualise boundary → evaluate → explain")

st.divider()

st.markdown("## Step 1: Inspect the dataset")

with st.expander("📘 Data dictionary", expanded=True):
    st.markdown(
        """
        | Column | Meaning |
        |---|---|
        | `feature_1` | First input feature |
        | `feature_2` | Second input feature |
        | `class_label` | Target class: 0 or 1 |
        """
    )

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Target", "class_label")

st.dataframe(df.head(20), use_container_width=True)
st.download_button("⬇️ Download dataset", df.to_csv(index=False), "knn_two_moons.csv", "text/csv")

st.divider()

st.markdown("## Step 2: Explore the pattern")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.scatter(df["feature_1"], df["feature_2"], c=df["class_label"], cmap="bwr", edgecolor="k", s=28, alpha=0.85)
ax.set_title("A curved pattern (KNN can adapt to non-linear shapes)")
ax.set_xlabel("feature_1")
ax.set_ylabel("feature_2")
ax.grid(True, alpha=0.2)
st.pyplot(fig)
plt.close(fig)

st.divider()

st.markdown("## Step 3: Split the data")

X = df[["feature_1", "feature_2"]]
y = df["class_label"]

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
    KNN “trains” by storing the training points. It then compares new points to these stored examples.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown("## Step 4: Fit the KNN classifier")

model = KNeighborsClassifier(n_neighbors=int(k), weights=str(weights), p=int(p))
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

gap = train_m["accuracy"] - test_m["accuracy"]
st.markdown(
    f"""
    <div class="success-box">
    <strong>Settings:</strong> k = {int(k)}, weights = {weights}, distance p = {int(p)}.<br><br>
    <strong>Overfitting check:</strong> training accuracy = {train_m['accuracy']:.3f}, testing accuracy = {test_m['accuracy']:.3f}
    (gap = {gap:+.3f}). Small k often increases overfitting.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What happens when k changes?"):
    st.markdown(
        """
        - **Small k** (e.g. 1–5): very local decisions, can follow noise, may overfit.
        - **Large k** (e.g. 25+): very smooth boundary, may miss real structure, may underfit.

        A good k is usually found by testing multiple values and choosing the best validation/test performance.
        """
    )

st.divider()

st.markdown("## Step 5: Visualise what KNN learned")
plot_decision_boundary(
    model=model,
    X_df=X_test,
    y_ser=y_test,
    title="Decision boundary on the test set (colour = predicted probability)",
)

st.divider()

st.markdown("## Step 6: Make a prediction")

with st.expander("Prediction explorer (choose one point)", expanded=True):
    c_left, c_right = st.columns(2)
    with c_left:
        x1_val = st.slider(
            "feature_1",
            float(df["feature_1"].min()) - 0.5,
            float(df["feature_1"].max()) + 0.5,
            0.0,
            step=0.01,
        )
    with c_right:
        x2_val = st.slider(
            "feature_2",
            float(df["feature_2"].min()) - 0.5,
            float(df["feature_2"].max()) + 0.5,
            0.0,
            step=0.01,
        )

    one = pd.DataFrame({"feature_1": [x1_val], "feature_2": [x2_val]})
    p1_class = float(model.predict_proba(one)[0, 1])
    pred_class = int(model.predict(one)[0])

    pcol1, pcol2 = st.columns(2)
    pcol1.metric("Predicted probability (class 1)", f"{p1_class:.3f}")
    pcol2.metric("Predicted class", str(pred_class))

    with st.expander("Show the nearest neighbours (concept)", expanded=False):
        st.markdown(
            """
            In a full KNN visualisation, you’d draw lines from the new point to its k closest training points.
            Here, the boundary plot already shows how the neighbourhood voting shapes the space.
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
            "Predicted 0": [tn, fn],
            "Predicted 1": [fp, tp],
        },
        index=["Actual 0", "Actual 1"],
    )
    st.dataframe(confusion_display, use_container_width=True)

with cm_col2:
    st.markdown(
        f"""
        <div class="info-box">
        <strong>Summary</strong><br><br>
        <strong>True positives:</strong> {tp}<br>
        <strong>False negatives:</strong> {fn}<br>
        <strong>False positives:</strong> {fp}<br>
        <strong>True negatives:</strong> {tn}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.markdown("## ⚠️ Limitations of KNN")
st.markdown(
    """
    <div class="warning-box">
    <strong>Important limitations:</strong><br><br>
    - Can be slow on large datasets (it compares to many stored points).<br>
    - Sensitive to feature scaling (one feature can dominate “distance”).<br>
    - The choice of k and distance metric matters a lot.<br>
    - Harder to interpret than a simple rule list, although the idea is intuitive.
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
        | Nearest neighbour | The most similar example in the dataset |
        | k | Number of neighbours used to vote |
        | Distance metric | How similarity is measured (e.g. Euclidean) |
        | Overfitting | When the model follows noise (often small k) |
        | Underfitting | When the model is too smooth (often very large k) |
        | Confusion matrix | Table of correct/incorrect classifications |
        | Precision | Of predicted positives, how many were correct |
        | Recall | Of actual positives, how many were found |
        """
    )

with tabs[1]:
    st.markdown(
        f"""
        This model uses two features (`feature_1`, `feature_2`) to predict a class label (0 or 1).

        KNN predicts a class by finding the **k closest training points** and voting.
        The current settings are:
        - k: `{int(k)}`
        - weights: `{weights}`
        - distance p: `{int(p)}`

        On the test set it achieved:
        - accuracy: `{test_m['accuracy']:.3f}`
        - precision: `{test_m['precision']:.3f}`
        - recall: `{test_m['recall']:.3f}`
        - F1: `{test_m['f1']:.3f}`

        The decision boundary plot shows how changing k can make the boundary more jagged (overfitting) or smoother (underfitting).
        """
    )

with tabs[2]:
    answer = f"""
This system uses supervised learning because it is trained using labelled examples where each record has input features and a known target class.

K-nearest neighbours (KNN) classifies new examples by comparing them to the k most similar training examples using a distance metric. The predicted class is chosen by a majority vote (or a distance-weighted vote).

The dataset was split into training and testing data. The training data is stored and used for comparisons, and the testing data is used to evaluate performance on unseen examples.

The model achieved an accuracy of {test_m['accuracy']:.3f}. However, accuracy alone can be misleading, so precision ({test_m['precision']:.3f}), recall ({test_m['recall']:.3f}) and the confusion matrix should also be used to understand different error types.

Overall, this is a {strength} model. Limitations include sensitivity to the choice of k and distance metric, possible slow performance on large datasets, and sensitivity to feature scaling. Therefore, KNN should be tuned and evaluated carefully before real use.
""".strip()

    st.markdown(answer)
    st.download_button(
        label="⬇️ Download HSC-style answer",
        data=answer,
        file_name="knn_hsc_answer.txt",
        mime="text/plain",
    )

with tabs[3]:
    st.markdown(
        """
        | Mistake | Why it is a problem | Better approach |
        |---|---|---|
        | Always choosing k=1 | Often overfits noise | Try multiple k values and compare test performance |
        | Ignoring scaling | Distances become unfair | Scale features in real-world datasets |
        | Only reporting accuracy | Hides important errors | Include precision/recall and confusion matrix |
        | Using an unsuitable distance metric | Changes what “similar” means | Match metric to the data and test results |
        """
    )

st.divider()

st.markdown("## 🧠 Student reflection")
st.markdown(
    """
    Answer these questions in your notes:

    1. What happens to the decision boundary when k increases?
    2. When does overfitting appear (train–test gap)?
    3. Does distance-weighted voting help when noise is high?
    4. Why can feature scaling be important for KNN?
    5. What trade-offs exist between interpretability and performance for KNN?
    """
)

