# loan_drl_governance_demo.py

import os
import random
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# 1. Config
# =========================================================

SEED = 42
DATA_PATH = "loan_approval_dataset.xlsx"
OUTPUT_DIR = "loan_drl_outputs"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. Utility
# =========================================================

def save_fig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=220, bbox_inches="tight")
    plt.close()


def decision_text(v: int) -> str:
    return "Approve" if int(v) == 1 else "Reject"


def add_bar_labels(ax, fmt: str = "{:.2f}", padding: int = 3) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            if np.isnan(height):
                labels.append("")
            else:
                labels.append(fmt.format(height))
        ax.bar_label(container, labels=labels, padding=padding, fontsize=9)


def save_dataframe_as_table_image(
    df: pd.DataFrame,
    filename: str,
    title: str,
    figsize: Tuple[int, int] = (16, 4),
    font_size: int = 9,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.5)

    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=220, bbox_inches="tight")
    plt.close()


# =========================================================
# 3. Load data
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


# =========================================================
# 4. EDA + feature engineering
# =========================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = ["Loan_Amount", "Annual_Income"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in add_features(): {missing}")

    df["loan_income_ratio"] = df["Loan_Amount"] / (df["Annual_Income"].clip(lower=1) * 5.0)

    # Only available on original historical dataset
    if "Approval_Status" in df.columns:
        df["actual_approved"] = (df["Approval_Status"] == "Approved").astype(int)

    return df


def plot_basic_eda(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    counts = df["Approval_Status"].value_counts()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Original Approval Status Distribution")
    plt.ylabel("Count")
    save_fig("01_original_approval_distribution.png")

    plt.figure(figsize=(7, 4))
    plt.hist(df["Credit_Score"], bins=30)
    plt.title("Credit Score Distribution")
    plt.xlabel("Credit Score")
    plt.ylabel("Frequency")
    save_fig("02_credit_score_distribution.png")

    plt.figure(figsize=(7, 5))
    for status in ["Approved", "Rejected"]:
        part = df[df["Approval_Status"] == status]
        plt.scatter(
            part["Annual_Income"],
            part["Loan_Amount"],
            alpha=0.5,
            label=status
        )
    plt.title("Annual Income vs Loan Amount")
    plt.xlabel("Annual Income")
    plt.ylabel("Loan Amount")
    plt.legend()
    save_fig("03_income_vs_loan.png")

    plt.figure(figsize=(7, 4))
    emp_counts = df["Employment_Status"].value_counts()
    plt.bar(emp_counts.index.astype(str), emp_counts.values)
    plt.title("Employment Status Distribution")
    plt.ylabel("Count")
    save_fig("04_employment_distribution.png")


# =========================================================
# 5. Human preference reward model
# =========================================================

EMPLOYMENT_SCORE: Dict[str, float] = {
    "Unemployed": 0.15,
    "Self-Employed": 0.55,
    "Employed": 0.85,
}

EDUCATION_SCORE: Dict[str, float] = {
    "High School": 0.45,
    "Bachelor": 0.65,
    "Master": 0.80,
    "PhD": 0.90,
}

PURPOSE_SCORE: Dict[str, float] = {
    "Business": 0.58,
    "Car": 0.60,
    "Education": 0.80,
    "Home": 0.72,
    "Personal": 0.50,
}


def build_human_preference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    credit_norm = (df["Credit_Score"] - 300) / 550
    affordability = 1 - np.minimum(df["loan_income_ratio"], 1.0)
    emp = df["Employment_Status"].map(EMPLOYMENT_SCORE).fillna(0.50)
    edu = df["Education"].map(EDUCATION_SCORE).fillna(0.60)
    purpose = df["Loan_Purpose"].map(PURPOSE_SCORE).fillna(0.58)

    df["pref_score"] = (
        0.50 * credit_norm
        + 0.20 * affordability
        + 0.15 * emp
        + 0.10 * edu
        + 0.05 * purpose
    )

    df["hard_reject"] = (
        (df["Credit_Score"] < 450) |
        (df["loan_income_ratio"] > 1.20)
    ).astype(int)

    df["human_pref"] = (
        (df["pref_score"] >= 0.58) &
        (df["hard_reject"] == 0)
    ).astype(int)

    return df


def build_biased_human_preference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demo-only biased human preference.
    This is intentionally artificial to show how human bias can alter outcomes.
    """
    df = df.copy()

    biased = df["human_pref"].copy()

    borderline_mask = (
        (df["pref_score"] >= 0.52) &
        (df["pref_score"] <= 0.65) &
        (df["hard_reject"] == 0)
    )

    male_borderline = borderline_mask & (df["Gender"].astype(str).str.lower() == "male")
    female_borderline = borderline_mask & (df["Gender"].astype(str).str.lower() == "female")

    biased.loc[male_borderline] = 1
    biased.loc[female_borderline] = 0

    df["biased_human_pref"] = biased.astype(int)
    return df


def plot_human_preference(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sample_df = df.sample(min(800, len(df)), random_state=SEED)

    for pref_value, label in [(0, "Human prefers Reject"), (1, "Human prefers Approve")]:
        part = sample_df[sample_df["human_pref"] == pref_value]
        plt.scatter(
            part["Credit_Score"],
            part["loan_income_ratio"],
            alpha=0.6,
            label=label,
        )

    plt.axvline(470, linestyle="--")
    plt.axhline(1.30, linestyle="--")
    plt.title("Human Preference Regions")
    plt.xlabel("Credit Score")
    plt.ylabel("Loan-to-Income Ratio")
    plt.legend()
    save_fig("05_human_preference_regions.png")


def plot_human_vs_drl(df: pd.DataFrame, drl_pred: np.ndarray) -> None:
    """
    Overlay human preference regions with DRL decisions using aligned test rows.
    """
    df_local = df.reset_index(drop=True).copy()
    if len(df_local) != len(drl_pred):
        raise ValueError("plot_human_vs_drl(): df and drl_pred must have the same length.")

    sample_size = min(800, len(df_local))
    sample_idx = df_local.sample(sample_size, random_state=SEED).index

    sample_df = df_local.loc[sample_idx].copy()
    drl_pred_sample = pd.Series(drl_pred, index=df_local.index).loc[sample_idx].values

    plt.figure(figsize=(8, 6))

    human_approve = sample_df["human_pref"] == 1
    human_reject = sample_df["human_pref"] == 0

    plt.scatter(
        sample_df.loc[human_approve, "Credit_Score"],
        sample_df.loc[human_approve, "loan_income_ratio"],
        alpha=0.55,
        label="Human Approve"
    )

    plt.scatter(
        sample_df.loc[human_reject, "Credit_Score"],
        sample_df.loc[human_reject, "loan_income_ratio"],
        alpha=0.55,
        label="Human Reject"
    )

    drl_approve = drl_pred_sample == 1
    drl_reject = drl_pred_sample == 0

    plt.scatter(
        sample_df.loc[drl_approve, "Credit_Score"],
        sample_df.loc[drl_approve, "loan_income_ratio"],
        facecolors="none",
        edgecolors="black",
        marker="o",
        s=70,
        linewidths=1.2,
        label="DRL Approve"
    )

    plt.scatter(
        sample_df.loc[drl_reject, "Credit_Score"],
        sample_df.loc[drl_reject, "loan_income_ratio"],
        marker="x",
        s=35,
        label="DRL Reject"
    )

    plt.axvline(470, linestyle="--")
    plt.axhline(1.30, linestyle="--")

    plt.title("Human Preference Regions with DRL Decisions")
    plt.xlabel("Credit Score")
    plt.ylabel("Loan-Income Ratio")
    plt.legend()
    save_fig("05_human_vs_drl_regions.png")


# =========================================================
# 6. DRL reward
# =========================================================

def reward_fn(
    action: np.ndarray,
    pref: np.ndarray,
    score: np.ndarray,
    hard_reject: np.ndarray,
    credit_score: np.ndarray,
    loan_income_ratio: np.ndarray,
) -> np.ndarray:
    align = np.where(action == pref, 1.0, -1.0)

    confidence = 0.5 + np.abs(score - 0.5) * 1.5
    reward = align * confidence

    risky_approval = (
        (action == 1) &
        (
            (hard_reject == 1) |
            (credit_score < 470) |
            (loan_income_ratio > 1.30)
        )
    )

    missed_good_case = (
        (action == 0) &
        (pref == 1) &
        (score > 0.70)
    )

    reward = reward - risky_approval * 1.50
    reward = reward - missed_good_case * 0.25

    return reward.astype(np.float32)


# =========================================================
# 7. Prepare state
# =========================================================

def prepare_state(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Governance choice: exclude Gender from state
    state_cols = [
        "Age",
        "Marital_Status",
        "Education",
        "Employment_Status",
        "Annual_Income",
        "Loan_Amount",
        "Loan_Purpose",
        "Credit_Score",
        "loan_income_ratio",
    ]

    X = pd.get_dummies(
        df[state_cols],
        columns=["Marital_Status", "Education", "Employment_Status", "Loan_Purpose"],
        drop_first=False,
    )
    return X, list(X.columns)


# =========================================================
# 8. DQN model
# =========================================================

class LoanQNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainOutput:
    model: nn.Module
    history: pd.DataFrame
    raw_pred: np.ndarray
    governed_pred: np.ndarray


def train_dqn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    score_train: np.ndarray,
    score_test: np.ndarray,
    hard_train: np.ndarray,
    hard_test: np.ndarray,
    credit_train: np.ndarray,
    credit_test: np.ndarray,
    lir_train: np.ndarray,
    lir_test: np.ndarray,
    epochs: int = 12,
) -> TrainOutput:
    model = LoanQNet(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    epsilon = 0.70
    history_rows = []

    for epoch in range(1, epochs + 1):
        with torch.no_grad():
            q_values = model(X_train_t).numpy()

        greedy_actions = q_values.argmax(axis=1)
        random_actions = np.random.randint(0, 2, size=len(X_train))
        explore = np.random.rand(len(X_train)) < epsilon
        actions = np.where(explore, random_actions, greedy_actions)

        rewards = reward_fn(
            actions,
            y_train,
            score_train,
            hard_train,
            credit_train,
            lir_train,
        )

        shuffled_idx = np.random.permutation(len(X_train))

        for start in range(0, len(shuffled_idx), 256):
            batch_idx = shuffled_idx[start:start + 256]

            states = torch.tensor(X_train[batch_idx], dtype=torch.float32)
            batch_actions = torch.tensor(actions[batch_idx], dtype=torch.int64)
            batch_rewards = torch.tensor(rewards[batch_idx], dtype=torch.float32)

            q_pred = model(states)
            q_target = q_pred.detach().clone()
            q_target[torch.arange(len(batch_idx)), batch_actions] = batch_rewards

            loss = loss_fn(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = max(0.05, epsilon * 0.85)

        with torch.no_grad():
            test_pred = model(X_test_t).argmax(dim=1).numpy()

        avg_reward = float(rewards.mean())
        acc = float((test_pred == y_test).mean())

        history_rows.append({
            "epoch": epoch,
            "avg_train_reward": avg_reward,
            "test_accuracy_vs_human_pref": acc,
        })

        print(f"Epoch {epoch:02d} | reward={avg_reward:.4f} | acc={acc:.4f}")

    with torch.no_grad():
        raw_pred = model(X_test_t).argmax(dim=1).numpy()

    governed_pred = np.where(
        (credit_test < 470) | (lir_test > 1.30),
        0,
        raw_pred
    )

    history_df = pd.DataFrame(history_rows)

    return TrainOutput(
        model=model,
        history=history_df,
        raw_pred=raw_pred,
        governed_pred=governed_pred,
    )


# =========================================================
# 9. Evaluation
# =========================================================

def evaluate_results(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    raw_pred: np.ndarray,
    governed_pred: np.ndarray,
) -> pd.DataFrame:
    raw_reward = reward_fn(
        raw_pred,
        y_test,
        df_test["pref_score"].values,
        df_test["hard_reject"].values,
        df_test["Credit_Score"].values,
        df_test["loan_income_ratio"].values,
    ).mean()

    gov_reward = reward_fn(
        governed_pred,
        y_test,
        df_test["pref_score"].values,
        df_test["hard_reject"].values,
        df_test["Credit_Score"].values,
        df_test["loan_income_ratio"].values,
    ).mean()

    summary = pd.DataFrame({
        "Metric": [
            "Original label accuracy vs human preference",
            "Raw DRL accuracy vs human preference",
            "Governed DRL accuracy vs human preference",
            "Human preference approval rate",
            "Raw DRL approval rate",
            "Governed DRL approval rate",
            "Raw DRL average reward",
            "Governed DRL average reward",
        ],
        "Value": [
            float((df_test["actual_approved"].values == y_test).mean()),
            float((raw_pred == y_test).mean()),
            float((governed_pred == y_test).mean()),
            float(y_test.mean()),
            float(raw_pred.mean()),
            float(governed_pred.mean()),
            float(raw_reward),
            float(gov_reward),
        ]
    })

    return summary


def plot_training(history_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["avg_train_reward"], marker="o")
    plt.title("Reward Trend During DQN Training")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    save_fig("06_reward_trend.png")

    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["test_accuracy_vs_human_pref"], marker="o")
    plt.title("Accuracy vs Human Preference During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    save_fig("07_training_accuracy.png")


def plot_comparison(summary_df: pd.DataFrame) -> None:
    metrics = [
        "Original label accuracy vs human preference",
        "Raw DRL accuracy vs human preference",
        "Governed DRL accuracy vs human preference",
    ]
    values = summary_df.set_index("Metric").loc[metrics, "Value"].values

    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.bar(["Original labels", "Raw DRL", "Governed DRL"], values)
    ax.set_title("Accuracy Against Human Preference")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.10)
    add_bar_labels(ax, fmt="{:.2f}")
    save_fig("08_accuracy_comparison.png")


def build_group_audit(
    df_test: pd.DataFrame,
    governed_pred: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    df_local = df_test.reset_index(drop=True).copy()
    rows = []

    for group in sorted(df_local[group_col].astype(str).unique()):
        mask = df_local[group_col].astype(str) == group
        rows.append({
            group_col: group,
            "Samples": int(mask.sum()),
            "Human preference approval rate": float(df_local.loc[mask, "human_pref"].mean()),
            "Biased human approval rate": float(df_local.loc[mask, "biased_human_pref"].mean()),
            "Governed DRL approval rate": float(governed_pred[mask].mean()),
        })

    return pd.DataFrame(rows)


def plot_combined_governance_audit(
    gender_audit_df: pd.DataFrame,
    education_audit_df: pd.DataFrame,
    purpose_audit_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    audit_configs = [
        (gender_audit_df, "Gender", axes[0], "Audit by Gender"),
        (education_audit_df, "Education", axes[1], "Audit by Education"),
        (purpose_audit_df, "Loan_Purpose", axes[2], "Audit by Loan Purpose"),
    ]

    for audit_df, group_col, ax, title in audit_configs:
        x = np.arange(len(audit_df))
        width = 0.25

        ax.bar(
            x - width,
            audit_df["Human preference approval rate"],
            width,
            label="Human preference"
        )
        ax.bar(
            x,
            audit_df["Biased human approval rate"],
            width,
            label="Biased human"
        )
        ax.bar(
            x + width,
            audit_df["Governed DRL approval rate"],
            width,
            label="Governed DRL"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(audit_df[group_col], rotation=20, ha="right")
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("Approval Rate")
        ax.set_title(title)

        add_bar_labels(ax, fmt="{:.2f}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "09_combined_governance_audit.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def build_bias_gap_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    bias_gap_df = audit_df.copy()

    bias_gap_df["Bias gap (Biased human - Human)"] = (
        bias_gap_df["Biased human approval rate"] - bias_gap_df["Human preference approval rate"]
    ).round(4)

    bias_gap_df["Governance gap (Governed DRL - Human)"] = (
        bias_gap_df["Governed DRL approval rate"] - bias_gap_df["Human preference approval rate"]
    ).round(4)

    return bias_gap_df[
        [
            "Gender",
            "Samples",
            "Human preference approval rate",
            "Biased human approval rate",
            "Governed DRL approval rate",
            "Bias gap (Biased human - Human)",
            "Governance gap (Governed DRL - Human)",
        ]
    ]


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Reject", "Approve"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("Actual Class", fontsize=11)
    ax.set_title("Confusion Matrix: Governed DRL vs Human Preference", fontsize=12, pad=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=90, va="center")

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_confusion_matrix_heatmap.png"), dpi=220, bbox_inches="tight")
    plt.close()


# =========================================================
# 10. New application comparison
# =========================================================

def prepare_new_applications() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Case": "A - strong profile",
            "Age": 36,
            "Gender": "Female",
            "Marital_Status": "Married",
            "Education": "Master",
            "Employment_Status": "Employed",
            "Annual_Income": 120000,
            "Loan_Amount": 180000,
            "Loan_Purpose": "Home",
            "Credit_Score": 790,
        },
        {
            "Case": "B - risky profile",
            "Age": 28,
            "Gender": "Male",
            "Marital_Status": "Single",
            "Education": "High School",
            "Employment_Status": "Unemployed",
            "Annual_Income": 35000,
            "Loan_Amount": 250000,
            "Loan_Purpose": "Personal",
            "Credit_Score": 430,
        },
        {
            "Case": "C - borderline male",
            "Age": 41,
            "Gender": "Male",
            "Marital_Status": "Married",
            "Education": "Bachelor",
            "Employment_Status": "Self-Employed",
            "Annual_Income": 80000,
            "Loan_Amount": 250000,
            "Loan_Purpose": "Business",
            "Credit_Score": 610,
        },
        {
            "Case": "D - borderline female",
            "Age": 41,
            "Gender": "Female",
            "Marital_Status": "Married",
            "Education": "Bachelor",
            "Employment_Status": "Self-Employed",
            "Annual_Income": 80000,
            "Loan_Amount": 250000,
            "Loan_Purpose": "Business",
            "Credit_Score": 610,
        },
        {
            "Case": "E - good credit but high burden",
            "Age": 33,
            "Gender": "Female",
            "Marital_Status": "Single",
            "Education": "Bachelor",
            "Employment_Status": "Employed",
            "Annual_Income": 70000,
            "Loan_Amount": 420000,
            "Loan_Purpose": "Home",
            "Credit_Score": 760,
        },
    ])


def score_new_applications(new_df: pd.DataFrame) -> pd.DataFrame:
    new_df = new_df.copy()
    new_df = add_features(new_df)
    new_df = build_human_preference(new_df)
    new_df = build_biased_human_preference(new_df)
    return new_df


def explain_case(row: pd.Series) -> str:
    reasons = []

    if row["Credit_Score"] >= 720:
        reasons.append("high credit score")
    elif row["Credit_Score"] < 470:
        reasons.append("very low credit score")
    else:
        reasons.append("mid-range credit score")

    if row["loan_income_ratio"] <= 0.70:
        reasons.append("manageable loan burden")
    elif row["loan_income_ratio"] > 1.30:
        reasons.append("loan burden too high")
    else:
        reasons.append("moderate loan burden")

    if row["Employment_Status"] == "Employed":
        reasons.append("stable employment")
    elif row["Employment_Status"] == "Unemployed":
        reasons.append("employment risk")
    else:
        reasons.append("self-employed / variable stability")

    reasons.append(f"pref_score={row['pref_score']:.2f}")

    if 0.52 <= row["pref_score"] <= 0.65:
        reasons.append("borderline case where human bias may appear")

    return ", ".join(reasons)


def predict_new_cases(
    model: nn.Module,
    new_df: pd.DataFrame,
    train_columns: List[str],
    train_means: pd.Series,
    train_stds: pd.Series,
) -> pd.DataFrame:
    X_new = pd.get_dummies(
        new_df[
            [
                "Age",
                "Marital_Status",
                "Education",
                "Employment_Status",
                "Annual_Income",
                "Loan_Amount",
                "Loan_Purpose",
                "Credit_Score",
                "loan_income_ratio",
            ]
        ],
        columns=["Marital_Status", "Education", "Employment_Status", "Loan_Purpose"],
        drop_first=False,
    )

    X_new = X_new.reindex(columns=train_columns, fill_value=0)
    X_new_scaled = ((X_new - train_means) / train_stds).astype(np.float32).values

    with torch.no_grad():
        raw = model(torch.tensor(X_new_scaled, dtype=torch.float32)).argmax(dim=1).numpy()

    governed = np.where(
        (new_df["Credit_Score"].values < 470) | (new_df["loan_income_ratio"].values > 1.30),
        0,
        raw
    )

    result = new_df[
        [
            "Case",
            "Gender",
            "Annual_Income",
            "Loan_Amount",
            "Credit_Score",
            "loan_income_ratio",
            "pref_score",
        ]
    ].copy()

    result["Human preference"] = [decision_text(v) for v in new_df["human_pref"].values]
    result["Biased human"] = [decision_text(v) for v in new_df["biased_human_pref"].values]
    result["DRL raw"] = [decision_text(v) for v in raw]
    result["DRL governed"] = [decision_text(v) for v in governed]
    result["Explanation"] = [explain_case(row) for _, row in new_df.iterrows()]

    return result


def plot_new_sample_decision_comparison(comparison_df: pd.DataFrame) -> None:
    chart_df = comparison_df.copy()

    mapping = {"Reject": 0, "Approve": 1}
    value_cols = ["Human preference", "Biased human", "DRL raw", "DRL governed"]

    for col in value_cols:
        chart_df[col] = chart_df[col].map(mapping)

    x = np.arange(len(chart_df))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - 1.5 * width, chart_df["Human preference"], width, label="Human preference")
    ax.bar(x - 0.5 * width, chart_df["Biased human"], width, label="Biased human")
    ax.bar(x + 0.5 * width, chart_df["DRL raw"], width, label="DRL raw")
    ax.bar(x + 1.5 * width, chart_df["DRL governed"], width, label="DRL governed")

    ax.set_xticks(x)
    ax.set_xticklabels(chart_df["Case"], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Reject (0)", "Approve (1)"])
    ax.set_ylabel("Decision")
    ax.set_title("Decision Comparison on New Sample Applications")
    ax.legend()

    add_bar_labels(ax, fmt="{:.0f}")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "11_new_sample_decision_comparison.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def save_final_comparison_table_chart(comparison_df: pd.DataFrame) -> None:
    display_df = comparison_df.copy()

    display_df = display_df.rename(
        columns={
            "Annual_Income": "Income",
            "Loan_Amount": "Loan",
            "Credit_Score": "Credit",
            "loan_income_ratio": "Loan/Income",
            "pref_score": "Pref Score",
            "Human preference": "Human",
            "Biased human": "Biased Human",
            "DRL raw": "DRL Raw",
            "DRL governed": "DRL Governed",
        }
    )

    numeric_cols = ["Income", "Loan", "Credit", "Loan/Income", "Pref Score"]
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:,.2f}" if isinstance(x, (int, float, np.integer, np.floating)) else x
            )

    if "Explanation" in display_df.columns:
        display_df["Explanation"] = display_df["Explanation"].apply(
            lambda x: "\n".join(textwrap.wrap(str(x), width=32))
        )

    fig, ax = plt.subplots(figsize=(24, 6))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.08)
        else:
            cell.set_height(0.16)

        if col == len(display_df.columns) - 1 and row > 0:
            cell.get_text().set_ha("left")

    plt.title("Final Comparison Table: New Sample Applications", fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "12_final_comparison_table.png"),
        dpi=240,
        bbox_inches="tight",
    )
    plt.close()


# =========================================================
# 11. Main
# =========================================================

def main() -> None:
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(df.shape)
    print(df.isna().sum())

    df = add_features(df)
    plot_basic_eda(df)

    df = build_human_preference(df)
    df = build_biased_human_preference(df)
    plot_human_preference(df)

    X, train_columns = prepare_state(df)

    split = train_test_split(
        X,
        df["human_pref"].values,
        df["pref_score"].values,
        df["hard_reject"].values,
        df["Credit_Score"].values,
        df["loan_income_ratio"].values,
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["human_pref"].values,
    )

    (
        X_train_df,
        X_test_df,
        y_train,
        y_test,
        score_train,
        score_test,
        hard_train,
        hard_test,
        credit_train,
        credit_test,
        lir_train,
        lir_test,
        df_train,
        df_test,
    ) = split

    train_means = X_train_df.mean()
    train_stds = X_train_df.std().replace(0, 1)

    X_train = ((X_train_df - train_means) / train_stds).astype(np.float32).values
    X_test = ((X_test_df - train_means) / train_stds).astype(np.float32).values

    print("Training DQN...")
    train_output = train_dqn(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        score_train=score_train,
        score_test=score_test,
        hard_train=hard_train,
        hard_test=hard_test,
        credit_train=credit_train,
        credit_test=credit_test,
        lir_train=lir_train,
        lir_test=lir_test,
        epochs=12,
    )

    df_test_reset = df_test.reset_index(drop=True)

    # Overlay aligned DRL decisions on human preference regions
    plot_human_vs_drl(df_test_reset, train_output.governed_pred)

    summary_df = evaluate_results(
        df_test=df_test_reset,
        y_test=y_test,
        raw_pred=train_output.raw_pred,
        governed_pred=train_output.governed_pred,
    )

    plot_training(train_output.history)
    plot_comparison(summary_df)

    gender_audit_df = build_group_audit(
        df_test=df_test_reset,
        governed_pred=train_output.governed_pred,
        group_col="Gender",
    )

    education_audit_df = build_group_audit(
        df_test=df_test_reset,
        governed_pred=train_output.governed_pred,
        group_col="Education",
    )

    purpose_audit_df = build_group_audit(
        df_test=df_test_reset,
        governed_pred=train_output.governed_pred,
        group_col="Loan_Purpose",
    )

    plot_combined_governance_audit(
        gender_audit_df=gender_audit_df,
        education_audit_df=education_audit_df,
        purpose_audit_df=purpose_audit_df,
    )

    bias_gap_df = build_bias_gap_summary(gender_audit_df)

    save_confusion_matrix(y_test, train_output.governed_pred)

    print("\n=== CLASSIFICATION REPORT: GOVERNED DRL VS HUMAN PREFERENCE ===")
    print(classification_report(y_test, train_output.governed_pred, digits=4))

    new_df = prepare_new_applications()
    new_df = score_new_applications(new_df)

    comparison_df = predict_new_cases(
        model=train_output.model,
        new_df=new_df,
        train_columns=train_columns,
        train_means=train_means,
        train_stds=train_stds,
    )

    plot_new_sample_decision_comparison(comparison_df)
    save_final_comparison_table_chart(comparison_df)

    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)
    gender_audit_df.to_csv(os.path.join(OUTPUT_DIR, "governance_audit_gender.csv"), index=False)
    education_audit_df.to_csv(os.path.join(OUTPUT_DIR, "governance_audit_education.csv"), index=False)
    purpose_audit_df.to_csv(os.path.join(OUTPUT_DIR, "governance_audit_purpose.csv"), index=False)
    bias_gap_df.to_csv(os.path.join(OUTPUT_DIR, "bias_gap_summary.csv"), index=False)
    train_output.history.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "new_application_comparison.csv"), index=False)

    save_dataframe_as_table_image(
        df=bias_gap_df.round(4),
        filename="13_bias_gap_summary_table.png",
        title="Bias Gap Summary Table",
        figsize=(20, 3.5),
        font_size=9,
    )

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\n=== GOVERNANCE AUDIT: GENDER ===")
    print(gender_audit_df.to_string(index=False))

    print("\n=== GOVERNANCE AUDIT: EDUCATION ===")
    print(education_audit_df.to_string(index=False))

    print("\n=== GOVERNANCE AUDIT: LOAN PURPOSE ===")
    print(purpose_audit_df.to_string(index=False))

    print("\n=== BIAS GAP SUMMARY ===")
    print(bias_gap_df.to_string(index=False))

    print("\n=== FINAL COMPARISON TABLE (NEW SAMPLE DATA) ===")
    print(comparison_df.to_string(index=False))

    print(f"\nAll charts and tables saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()