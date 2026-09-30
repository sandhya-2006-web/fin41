import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/raw/bank_transactions.json")


def load_transactions():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def build_features(df):
    df = df.copy()

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------
    df["value_date"] = pd.to_datetime(df["value_date"])
    df["posted_date"] = pd.to_datetime(df["posted_date"])

    df["day_of_week"] = df["value_date"].dt.dayofweek
    df["day_of_month"] = df["value_date"].dt.day
    df["month"] = df["value_date"].dt.month

    # ---------------------------------------------------------
    # Transaction amount
    # ---------------------------------------------------------
    df["amount"] = df["debit"] + df["credit"]

    df["signed_amount"] = (
        df["credit"] - df["debit"]
    )

    df["is_debit"] = (
        df["debit"] > 0
    ).astype(int)

    df["is_credit"] = (
        df["credit"] > 0
    ).astype(int)

    # ---------------------------------------------------------
    # Account-level statistics
    # ---------------------------------------------------------
    account_group = df.groupby("account_id")

    df["account_transaction_count"] = (
        account_group["id"]
        .transform("count")
    )

    df["account_mean_amount"] = (
        account_group["amount"]
        .transform("mean")
    )

    df["account_median_amount"] = (
        account_group["amount"]
        .transform("median")
    )

    df["account_std_amount"] = (
        account_group["amount"]
        .transform("std")
        .fillna(0)
    )

    df["account_max_amount"] = (
        account_group["amount"]
        .transform("max")
    )

    # ---------------------------------------------------------
    # Amount deviation from account behavior
    # ---------------------------------------------------------
    df["amount_zscore"] = (
        (
            df["amount"]
            - df["account_mean_amount"]
        )
        / df["account_std_amount"].replace(0, np.nan)
    ).fillna(0)

    df["amount_vs_account_median"] = (
        df["amount"]
        / df["account_median_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---------------------------------------------------------
    # Daily transaction velocity
    # ---------------------------------------------------------
    df["daily_transaction_count"] = (
        df.groupby(
            ["account_id", "value_date"]
        )["id"]
        .transform("count")
    )

    # ---------------------------------------------------------
    # Counterparty frequency
    # ---------------------------------------------------------
    counterparty = (
        df["counterparty_text"]
        .fillna("UNKNOWN")
    )

    df["counterparty_frequency"] = (
        counterparty.map(
            counterparty.value_counts()
        )
    )

    # ---------------------------------------------------------
    # Narration frequency
    # ---------------------------------------------------------
    df["narration_frequency"] = (
        df["raw_narration"]
        .map(df["raw_narration"].value_counts())
    )

    # ---------------------------------------------------------
    # Bank reference presence
    # ---------------------------------------------------------
    df["has_bank_ref"] = (
        df["bank_ref"].notna()
    ).astype(int)

    # ---------------------------------------------------------
    # Cheque presence
    # ---------------------------------------------------------
    df["has_cheque"] = (
        df["cheque_no"].notna()
    ).astype(int)

    # ---------------------------------------------------------
    # Balance consistency
    # ---------------------------------------------------------
    df = df.sort_values(
        ["account_id", "value_date", "line_no"]
    ).copy()

    df["previous_balance"] = (
        df.groupby("account_id")["running_balance"]
        .shift(1)
    )

    df["expected_balance_change"] = (
        df["credit"] - df["debit"]
    )

    df["observed_balance_change"] = (
        df["running_balance"]
        - df["previous_balance"]
    )

    df["balance_difference"] = (
        df["observed_balance_change"]
        - df["expected_balance_change"]
    )

    df["balance_mismatch"] = (
        df["previous_balance"].notna()
        & (
            df["balance_difference"].abs()
            >= 0.01
        )
    ).astype(int)

    # ---------------------------------------------------------
    # Clean infinite values
    # ---------------------------------------------------------
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill numerical missing values
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numerical_columns] = (
        df[numerical_columns]
        .fillna(0)
    )

    return df


def get_model_features(df):
    """
    Return numerical features for the anomaly model.
    """

    feature_columns = [
        "amount",
        "signed_amount",
        "is_debit",
        "is_credit",
        "day_of_week",
        "day_of_month",
        "month",
        "account_transaction_count",
        "account_mean_amount",
        "account_median_amount",
        "account_std_amount",
        "account_max_amount",
        "amount_zscore",
        "amount_vs_account_median",
        "daily_transaction_count",
        "counterparty_frequency",
        "narration_frequency",
        "has_bank_ref",
        "has_cheque",
        "balance_difference",
        "balance_mismatch",
    ]

    return df[feature_columns].copy()


if __name__ == "__main__":
    df = load_transactions()

    features = build_features(df)

    model_features = get_model_features(features)

    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    print("\nTransactions:", len(features))

    print("\nModel features:")
    for column in model_features.columns:
        print(" -", column)

    print("\nFeature matrix shape:")
    print(model_features.shape)

    print("\nBalance mismatches:")
    print(
        features["balance_mismatch"].sum()
    )

    print("\nSample engineered features:")
    print(
        model_features.head(5).to_string(
            index=False
        )
    )