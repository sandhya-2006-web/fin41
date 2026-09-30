import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/bank_transactions.json")


def main():
    print("=" * 70)
    print("FIN-41 TRANSACTION ANALYSIS")
    print("=" * 70)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------
    df["value_date"] = pd.to_datetime(df["value_date"])
    df["posted_date"] = pd.to_datetime(df["posted_date"])

    # ---------------------------------------------------------
    # Basic transaction features
    # ---------------------------------------------------------
    df["amount"] = df["debit"] + df["credit"]
    df["signed_amount"] = df["credit"] - df["debit"]

    df["is_debit"] = (df["debit"] > 0).astype(int)
    df["is_credit"] = (df["credit"] > 0).astype(int)

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------
    print("\nTOTAL TRANSACTIONS")
    print("------------------")
    print(len(df))

    print("\nUNIQUE ACCOUNTS")
    print("----------------")
    print(df["account_id"].nunique())

    print("\nDATE RANGE")
    print("----------")
    print("From:", df["value_date"].min())
    print("To  :", df["value_date"].max())

    print("\nTRANSACTION AMOUNT")
    print("------------------")
    print(df["amount"].describe())

    # ---------------------------------------------------------
    # Debit / credit
    # ---------------------------------------------------------
    print("\nDEBIT TRANSACTIONS")
    print("------------------")
    print((df["debit"] > 0).sum())

    print("\nCREDIT TRANSACTIONS")
    print("-------------------")
    print((df["credit"] > 0).sum())

    # ---------------------------------------------------------
    # Largest transactions
    # ---------------------------------------------------------
    print("\nTOP 10 LARGEST TRANSACTIONS")
    print("----------------------------")

    largest = df.nlargest(10, "amount")[
        [
            "id",
            "account_id",
            "value_date",
            "debit",
            "credit",
            "amount",
            "running_balance",
            "raw_narration",
        ]
    ]

    print(largest.to_string(index=False))

    # ---------------------------------------------------------
    # Account activity
    # ---------------------------------------------------------
    print("\nTOP ACCOUNTS BY TRANSACTION COUNT")
    print("----------------------------------")

    account_counts = (
        df.groupby("account_id")
        .size()
        .sort_values(ascending=False)
        .head(20)
    )

    print(account_counts.to_string())

    # ---------------------------------------------------------
    # Account amount statistics
    # ---------------------------------------------------------
    print("\nACCOUNT AMOUNT STATISTICS")
    print("-------------------------")

    account_stats = (
        df.groupby("account_id")["amount"]
        .agg(
            transaction_count="count",
            total_amount="sum",
            average_amount="mean",
            max_amount="max",
        )
        .sort_values("transaction_count", ascending=False)
        .head(20)
    )

    print(account_stats.to_string())

    # ---------------------------------------------------------
    # Same-day transaction velocity
    # ---------------------------------------------------------
    df["transactions_same_day"] = (
        df.groupby(["account_id", "value_date"])["id"]
        .transform("count")
    )

    print("\nHIGHEST ACCOUNT DAILY TRANSACTION COUNTS")
    print("-----------------------------------------")

    print(
        df[
            [
                "account_id",
                "value_date",
                "transactions_same_day",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "transactions_same_day",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Counterparties
    # ---------------------------------------------------------
    print("\nTOP COUNTERPARTIES")
    print("------------------")

    print(
        df["counterparty_text"]
        .value_counts(dropna=True)
        .head(20)
        .to_string()
    )

    # ---------------------------------------------------------
    # Narration
    # ---------------------------------------------------------
    print("\nTOP NARRATION PATTERNS")
    print("----------------------")

    print(
        df["raw_narration"]
        .value_counts()
        .head(20)
        .to_string()
    )

    # ---------------------------------------------------------
    # Balance consistency
    # ---------------------------------------------------------
    print("\nBALANCE CHANGE CHECK")
    print("--------------------")

    df_sorted = df.sort_values(
        ["account_id", "value_date", "line_no"]
    ).copy()

    df_sorted["previous_balance"] = (
        df_sorted.groupby("account_id")["running_balance"]
        .shift(1)
    )

    df_sorted["observed_balance_change"] = (
        df_sorted["running_balance"]
        - df_sorted["previous_balance"]
    )

    df_sorted["expected_balance_change"] = (
        df_sorted["credit"]
        - df_sorted["debit"]
    )

    df_sorted["balance_difference"] = (
        df_sorted["observed_balance_change"]
        - df_sorted["expected_balance_change"]
    )

    balance_check = df_sorted[
        df_sorted["previous_balance"].notna()
    ].copy()

    print(
        "Transactions with previous balance:",
        len(balance_check),
    )

    print(
        "Exact balance matches:",
        (
            balance_check["balance_difference"].abs() < 0.01
        ).sum(),
    )

    print(
        "Balance mismatches:",
        (
            balance_check["balance_difference"].abs() >= 0.01
        ).sum(),
    )


if __name__ == "__main__":
    main()
