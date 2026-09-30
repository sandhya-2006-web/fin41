import pandas as pd


def generate_reason(row):
    reasons = []

    # Large relative to account behavior
    if abs(row["amount_zscore"]) >= 3:
        reasons.append(
            "Amount is extremely unusual for this account"
        )
    elif abs(row["amount_zscore"]) >= 2:
        reasons.append(
            "Amount is unusually high for this account"
        )

    # Balance inconsistency
    if row["balance_mismatch"] == 1:
        reasons.append(
            "Running balance does not match expected transaction change"
        )

    # High daily activity
    if row["daily_transaction_count"] >= 8:
        reasons.append(
            "High transaction activity for this account on the same day"
        )
    elif row["daily_transaction_count"] >= 5:
        reasons.append(
            "Elevated transaction activity for this account on the same day"
        )

    # Rare counterparty
    if row["counterparty_frequency"] <= 2:
        reasons.append(
            "Counterparty is rarely seen in the dataset"
        )

    # Rare narration
    if row["narration_frequency"] <= 1:
        reasons.append(
            "Transaction narration pattern is rarely seen"
        )

    # Missing reference
    if row["has_bank_ref"] == 0:
        reasons.append(
            "Bank reference is missing"
        )

    if not reasons:
        reasons.append(
            "Transaction is unusual compared with learned transaction patterns"
        )

    return "; ".join(reasons)


def main():
    input_path = "data/anomaly_results.csv"
    output_path = "data/anomaly_results.csv"

    df = pd.read_csv(input_path)

    # Only generate detailed explanations for flagged
    # transactions. Normal transactions receive a generic reason.
    df["anomaly_reason"] = df.apply(
        generate_reason,
        axis=1,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("=" * 70)
    print("ANOMALY EXPLANATIONS")
    print("=" * 70)

    print("\nTop anomalous transactions with reasons:\n")

    columns = [
        "id",
        "account_id",
        "amount",
        "risk_score",
        "risk_level",
        "anomaly_reason",
    ]

    print(
        df.head(20)[columns].to_string(
            index=False
        )
    )

    print("\nUpdated:")
    print(output_path)


if __name__ == "__main__":
    main()