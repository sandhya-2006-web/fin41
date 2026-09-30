import pandas as pd


def generate_reason(row):
    reasons = []
    
    narration = str(row["raw_narration"]).upper()

    if "OWN A/C" in narration or "OWN A/C TRF" in narration or "TRF/OWN" in narration:
        reasons.append(
            "Appears to be a transfer between own accounts"
        )

    # Amount behavior
    if abs(row["amount_zscore"]) >= 3:
        reasons.append(
            "Amount is extremely unusual for this account"
        )
    elif abs(row["amount_zscore"]) >= 2:
        reasons.append(
            "Amount is unusually high for this account"
        )

    # Balance consistency
    if row["balance_mismatch"] == 1:
        reasons.append(
            "Running balance does not match expected transaction change"
        )

    # Transaction velocity
    if row["daily_transaction_count"] >= 8:
        reasons.append(
            "High transaction activity for this account on the same day"
        )
    elif row["daily_transaction_count"] >= 5:
        reasons.append(
            "Elevated transaction activity for this account on the same day"
        )

    # Counterparty frequency
    if row["counterparty_frequency"] <= 2:
        reasons.append(
            "Counterparty is rarely seen in the dataset"
        )

    # Narration frequency
    if row["narration_frequency"] <= 1:
        reasons.append(
            "Transaction narration pattern is rarely seen"
        )

    # Bank reference
    if row["has_bank_ref"] == 0:
        reasons.append(
            "Bank reference is missing"
        )

    if not reasons:
        reasons.append(
            "Transaction is unusual compared with learned transaction patterns"
        )

    return "; ".join(reasons)
def classify_transaction(row):

    narration = str(row["raw_narration"]).upper()

    if (
        "OWN A/C" in narration
        or "OWN A/C TRF" in narration
        or "TRF/OWN" in narration
    ):
        return "OWN_ACCOUNT_TRANSFER"

    if row["risk_level"] == "HIGH":
        return "HIGH_ANOMALY"

    if row["risk_level"] == "MEDIUM":
        return "MEDIUM_ANOMALY"

    return "NORMAL"


def main():

    input_path = "data/anomaly_results.csv"

    df = pd.read_csv(input_path)

    df["anomaly_reason"] = df.apply(
        generate_reason,
        axis=1
    )
    df["transaction_class"] = df.apply(
    classify_transaction,
    axis=1
    )

    # Put the most anomalous transactions first
    df = df.sort_values(
        "risk_score",
        ascending=False
    )

    df.to_csv(
        input_path,
        index=False
    )

    print("=" * 70)
    print("ANOMALY EXPLANATIONS")
    print("=" * 70)

    print("\nTop 20 transactions:\n")

    columns = [
        "id",
        "account_id",
        "amount",
        "risk_score",
        "risk_level",
        "anomaly_reason"
    ]

    print(
        df.head(20)[columns].to_string(
            index=False
        )
    )

    print("\nUpdated:")
    print(input_path)


if __name__ == "__main__":
    main()