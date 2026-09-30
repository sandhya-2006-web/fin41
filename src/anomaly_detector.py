import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import load_transactions, build_features, get_model_features


def main():
    print("=" * 70)
    print("FIN-41 ANOMALY DETECTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load and engineer features
    # ---------------------------------------------------------
    df = load_transactions()
    df = build_features(df)

    X = get_model_features(df)

    print("\nTransactions:", len(df))
    print("Features:", X.shape[1])

    # ---------------------------------------------------------
    # Scale features
    # ---------------------------------------------------------
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # Isolation Forest
    # ---------------------------------------------------------
    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    df["anomaly_prediction"] = model.predict(X_scaled)

    # Isolation Forest:
    # -1 = anomaly
    #  1 = normal

    df["is_anomaly"] = (
        df["anomaly_prediction"] == -1
    ).astype(int)

    # decision_function:
    # higher = more normal
    # lower = more anomalous

    df["anomaly_score"] = (
        -model.decision_function(X_scaled)
    )

    # ---------------------------------------------------------
    # Percentile-based risk score
    # ---------------------------------------------------------
    df["risk_score"] = (
        df["anomaly_score"]
        .rank(pct=True)
        * 100
    )

    # ---------------------------------------------------------
    # Risk categories
    # ---------------------------------------------------------
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[
            -float("inf"),
            80,
            95,
            float("inf"),
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
        ],
    )

    output_columns = [
    # Original transaction information
    "id",
    "account_id",
    "value_date",
    "posted_date",
    "debit",
    "credit",
    "amount",
    "running_balance",
    "raw_narration",
    "counterparty_text",
    "bank_ref",
    "cheque_no",

    # Engineered anomaly signals
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

    # Model output
    "anomaly_score",
    "risk_score",
    "risk_level",
    "is_anomaly",
]

    output = df[output_columns].copy()

    output = output.sort_values(
        "risk_score",
        ascending=False,
    )

    output_path = "data/anomaly_results.csv"

    output.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\nANOMALY SUMMARY")
    print("----------------")

    print(
        "Total transactions:",
        len(output),
    )

    print(
        "Anomalies detected:",
        output["is_anomaly"].sum(),
    )

    print(
        "Anomaly percentage:",
        round(
            output["is_anomaly"].mean() * 100,
            2,
        ),
        "%",
    )

    print("\nRISK LEVEL COUNTS")
    print("-----------------")

    print(
        output["risk_level"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Top anomalies
    # ---------------------------------------------------------
    print("\nTOP 20 ANOMALOUS TRANSACTIONS")
    print("------------------------------")

    top_anomalies = output.head(20)[
        [
            "id",
            "account_id",
            "value_date",
            "amount",
            "debit",
            "credit",
            "balance_mismatch",
            "amount_zscore",
            "daily_transaction_count",
            "risk_score",
            "risk_level",
            "raw_narration",
        ]
    ]

    print(
        top_anomalies.to_string(
            index=False
        )
    )

    print("\nSaved results to:")
    print(output_path)


if __name__ == "__main__":
    main()