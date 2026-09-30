import pandas as pd

INPUT_PATH = "data/anomaly_results.csv"
OUTPUT_PATH = "data/anomaly_results_explained.csv"


def explain(row):
    reasons = []

    # Large transaction
    if row["amount_zscore"] >= 2:
        reasons.append("Unusually large transaction")

    # Compared with account history
    if row["amount_vs_account_median"] >= 3:
        reasons.append("Amount is much higher than account median")

    # Many transactions on same day
    if row["daily_transaction_count"] >= 5:
        reasons.append("High transaction activity on same day")

    # Balance inconsistency
    if row["balance_mismatch"] == 1:
        reasons.append("Balance mismatch")

    # Rare counterparty
    if row["counterparty_frequency"] <= 2:
        reasons.append("Rare counterparty")

    # Rare narration
    if row["narration_frequency"] <= 2:
        reasons.append("Rare transaction narration")

    # Missing references
    if row["has_bank_ref"] == 0:
        reasons.append("Missing bank reference")

    if not reasons:
        reasons.append("Multiple unusual transaction characteristics")

    return "; ".join(reasons)


def main():
    df = pd.read_csv(INPUT_PATH)

    # Generate explanation
    df["anomaly_reason"] = df.apply(explain, axis=1)

    # Save complete results
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("ANOMALY EXPLANATION")
    print("=" * 70)

    print("\nTotal transactions:", len(df))
    print("Anomalies:", df["is_anomaly"].sum())

    print("\nTOP ANOMALIES WITH EXPLANATIONS")
    print("-" * 70)

    anomalies = (
        df[df["is_anomaly"] == 1]
        .sort_values("risk_score", ascending=False)
        .head(20)
    )

    columns = [
        "id",
        "account_id",
        "value_date",
        "amount",
        "risk_score",
        "risk_level",
        "anomaly_reason",
        "raw_narration",
    ]

    print(
        anomalies[columns].to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()