"""
Neural Networks — HSC Learning Page
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

st.set_page_config(page_title="Neural Networks", page_icon="🧠", layout="wide")

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

    X = np.vstack([np.c_[x0, y0], np.c_[x1, y1]])
    y = np.array([0] * n0 + [1] * n1)

    X = X + rng.normal(0, noise, size=X.shape)

    df = pd.DataFrame(
        {
            "feature_1": np.round(X[:, 0], 3),
            "feature_2": np.round(X[:, 1], 3),
            "class_label": y.astype(int),
        }
    )
    return df


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_decision_boundary(model: MLPClassifier, X: pd.DataFrame, y: pd.Series, title: str) -> None:
    x_min, x_max = float(X["feature_1"].min()) - 0.5, float(X["feature_1"].max()) + 0.5
    y_min, y_max = float(X["feature_2"].min()) - 0.5, float(X["feature_2"].max()) + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = pd.DataFrame({"feature_1": xx.ravel(), "feature_2": yy.ravel()})

    proba = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    contour = ax.contourf(xx, yy, proba, levels=30, cmap="RdYlBu", alpha=0.85)
    fig.colorbar(contour, ax=ax, label="P(class 1)")

    ax.scatter(
        X["feature_1"],
        X["feature_2"],
        c=y,
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


st.markdown('<div class="main-title">🧠 Neural Networks</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Learn a flexible model that can capture non-linear patterns.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    <strong>Scenario:</strong> You are building a model to classify two groups of students from two measurements.
    The groups overlap in a curved way, so a straight-line classifier is often not enough.
    A neural network can learn a non-linear boundary.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("What is a neural network?", expanded=True):
    st.markdown(
        """
        A neural network (multi-layer perceptron) is made of:
        - **inputs** (your features),
        - one or more **hidden layers** of **neurons**,
        - an **output** that predicts a class (or probability).

        Each neuron computes a weighted sum of inputs, then applies an **activation function**.
        Stacking layers creates a flexible model that can learn non-linear patterns.
        """
    )

with st.sidebar:
    st.header("⚙️ Controls")

    n_samples_ctl = st.slider("Number of points", 100, 2000, 500, step=50)
    noise_ctl = st.slider("Noise level", 0.00, 0.40, 0.10, step=0.01)
    test_size_ctl = st.slider("Test data percentage", 0.10, 0.50, 0.25, step=0.05)
    seed_ctl = st.number_input("Random seed", min_value=0, value=17, step=1)

    st.subheader("Network settings")
    n_layers = st.slider("Hidden layers", 1, 3, 2, step=1)
    neurons_per_layer = st.slider("Neurons per hidden layer", 2, 64, 16, step=2)
    activation = st.selectbox("Activation", ["relu", "tanh", "logistic"])
    alpha = st.slider(
        "Regularisation (alpha)",
        0.0000,
        0.0500,
        0.0010,
        step=0.0005,
        help="Higher alpha usually reduces overfitting but can underfit.",
    )
    max_iter = st.slider("Training iterations (max_iter)", 50, 1500, 400, step=50)

    with st.expander("🎚️ What do these controls do?"):
        st.markdown(
            """
            **Noise level** makes the classes harder to separate.  
            **Hidden layers / neurons** control model flexibility.  
            **Activation** changes how neurons create non-linearity.  
            **Alpha** adds regularisation to reduce overfitting.  
            **max_iter** controls how long the optimiser trains.
            """
        )

df = generate_two_moons_dataset(
    n_samples=int(n_samples_ctl),
    noise=float(noise_ctl),
    seed=int(seed_ctl),
)

st.markdown("## 🧭 Machine learning process")
st.markdown("Inspect data → choose features and target → split data → train network → visualise boundary → evaluate → explain")

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
st.download_button("⬇️ Download dataset", df.to_csv(index=False), "neural_network_two_moons.csv", "text/csv")

with st.expander("Class balance", expanded=True):
    balance = df["class_label"].value_counts().rename(index={0: "class 0", 1: "class 1"}).to_frame("count")
    st.dataframe(balance, use_container_width=True)

st.divider()

st.markdown("## Step 2: Explore the pattern")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.scatter(df["feature_1"], df["feature_2"], c=df["class_label"], cmap="bwr", edgecolor="k", s=28, alpha=0.85)
ax.set_title("The classes form a curved pattern (non-linear)")
ax.set_xlabel("feature_1")
ax.set_ylabel("feature_2")
ax.grid(True, alpha=0.2)
st.pyplot(fig)
plt.close(fig)

with st.expander("Why is this hard for a straight line?"):
    st.markdown(
        """
        If the classes wrap around each other, no single straight line can separate them well.
        Neural networks can build a non-linear boundary using hidden layers.
        """
    )

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
    The training data is used to learn the weights. The testing data is kept unseen to measure generalisation.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown("## Step 4: Train the neural network")

hidden_layer_sizes = tuple([int(neurons_per_layer)] * int(n_layers))

model = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=str(activation),
    alpha=float(alpha),
    max_iter=int(max_iter),
    solver="adam",
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

gap = train_m["accuracy"] - test_m["accuracy"]
st.markdown(
    f"""
    <div class="success-box">
    <strong>Architecture:</strong> hidden layers = {int(n_layers)}, neurons/layer = {int(neurons_per_layer)}, activation = {activation}.<br><br>
    <strong>Overfitting check:</strong> training accuracy = {train_m['accuracy']:.3f}, testing accuracy = {test_m['accuracy']:.3f}
    (gap = {gap:+.3f}). Large gaps can mean the model is too complex for the data.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Training details (loss curve)", expanded=False):
    if hasattr(model, "loss_curve_") and len(model.loss_curve_) > 1:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(model.loss_curve_)
        ax.set_title("Training loss over iterations")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.markdown("Loss curve not available for this run.")

st.divider()

st.markdown("## Step 5: Visualise what the network learned")
plot_decision_boundary(
    model=model,
    X=X_test,
    y=y_test,
    title="Decision boundary on the test set (colour = predicted probability)",
)

with st.expander("What should you look for?"):
    st.markdown(
        """
        A good model will create a boundary that curves between the two moons without being overly jagged.
        If the boundary is too wiggly, it may be overfitting.
        """
    )

st.divider()

st.markdown("## Step 6: Make a prediction")

with st.expander("Prediction explorer (choose one point)", expanded=True):
    c_left, c_right = st.columns(2)
    with c_left:
        x1 = st.slider("feature_1", float(df["feature_1"].min()) - 0.5, float(df["feature_1"].max()) + 0.5, 0.0, step=0.01)
    with c_right:
        x2 = st.slider("feature_2", float(df["feature_2"].min()) - 0.5, float(df["feature_2"].max()) + 0.5, 0.0, step=0.01)

    one = pd.DataFrame({"feature_1": [x1], "feature_2": [x2]})
    p = float(model.predict_proba(one)[0, 1])
    pred_class = int(model.predict(one)[0])

    p1, p2 = st.columns(2)
    p1.metric("Predicted probability (class 1)", f"{p:.3f}")
    p2.metric("Predicted class", str(pred_class))

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

st.markdown(
    """
    <div class="warning-box">
    Neural networks can be powerful, but they can also overfit if they are too complex, or if the dataset is small/noisy.
    Always evaluate on unseen data.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown("## ⚠️ Limitations of neural networks")
st.markdown(
    """
    <div class="warning-box">
    <strong>Important limitations:</strong><br><br>
    - Can overfit if the network is too large for the data.<br>
    - Often less interpretable than simpler models (harder to explain decisions).<br>
    - Training can be sensitive to settings (architecture, regularisation, iterations).<br>
    - Real-world systems must consider data quality, bias, privacy, and fairness.
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
        | Neuron | A unit that combines inputs using weights then applies an activation |
        | Layer | A group of neurons (input, hidden, output) |
        | Activation function | Non-linear function that helps learn complex patterns |
        | Weights | Learned values that control the influence of each input |
        | Training | Adjusting weights to reduce error on labelled examples |
        | Overfitting | When performance is good on training data but worse on new data |
        | Generalisation | How well the model works on unseen data |
        | Confusion matrix | Table of correct/incorrect classifications |
        """
    )

with tabs[1]:
    st.markdown(
        f"""
        This model uses two features (`feature_1`, `feature_2`) to predict a class label (0 or 1).
        The classes form a curved pattern, so a non-linear model is useful.

        The neural network used:
        - hidden layers: `{int(n_layers)}`
        - neurons per layer: `{int(neurons_per_layer)}`
        - activation: `{activation}`
        - regularisation alpha: `{float(alpha):.4f}`

        On the test set it achieved:
        - accuracy: `{test_m['accuracy']:.3f}`
        - precision: `{test_m['precision']:.3f}`
        - recall: `{test_m['recall']:.3f}`
        - F1: `{test_m['f1']:.3f}`

        The decision boundary plot shows how the network learned a non-linear separation between the two classes.
        """
    )

with tabs[2]:
    answer = f"""
This system uses supervised learning because it is trained using labelled examples, where each record includes input features and a known target class.

A neural network is suitable because the relationship between the inputs and the classes is non-linear, meaning a simple straight-line boundary may not separate the classes well. The network uses layers of neurons and activation functions to learn complex patterns by adjusting weights during training.

The dataset was split into training and testing data. The model learned from the training data and was evaluated on unseen testing data to measure generalisation. The model achieved an accuracy of {test_m['accuracy']:.3f}. Precision was {test_m['precision']:.3f} and recall was {test_m['recall']:.3f}, which help describe different types of classification error. The confusion matrix also shows true positives, true negatives, false positives and false negatives.

Overall, this is a {strength} model. However, it has limitations such as overfitting if the network is too large for the data and reduced interpretability compared with simpler models. Ethical considerations such as bias, privacy and fairness must also be considered in real applications.
""".strip()

    st.markdown(answer)
    st.download_button(
        label="⬇️ Download HSC-style answer",
        data=answer,
        file_name="neural_network_hsc_answer.txt",
        mime="text/plain",
    )

with tabs[3]:
    st.markdown(
        """
        | Mistake | Why it is a problem | Better approach |
        |---|---|---|
        | Treating neural networks as “magic” | They still depend on data quality and settings | Explain training, evaluation, and limitations |
        | Only using accuracy | Accuracy can hide important mistakes | Report precision/recall and confusion matrix |
        | Making the network very large | Can overfit, especially on small/noisy data | Use regularisation and test performance |
        | Ignoring interpretability/ethics | Real decisions need explanation and fairness | Mention bias, privacy, transparency, oversight |
        """
    )

st.divider()

st.markdown("## 🧠 Student reflection")
st.markdown(
    """
    Answer these questions in your notes:

    1. What happens when you increase layers or neurons? When does overfitting appear?
    2. How does noise level affect the decision boundary and metrics?
    3. Which activation function works best on this dataset? Why might that be?
    4. Why can neural networks be less interpretable than decision trees?
    5. What ethical issues could arise if a neural network were used on real student data?
    """
)

