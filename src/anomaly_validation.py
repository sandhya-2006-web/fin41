import pandas as pd


INPUT_PATH = "data/anomaly_results.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    # Identify likely own-account transfers from narration
    own_account = (
        df["raw_narration"]
        .fillna("")
        .str.contains(
            r"OWN A/C|OWN A/C TRF|OWN",
            case=False,
            regex=True
        )
    )

    # Identify high-risk transactions
    high_risk = df["risk_level"].eq("HIGH")

    # Transactions that are both HIGH risk and look like
    # own-account transfers
    own_account_high_risk = df[own_account & high_risk].copy()

    print("=" * 70)
    print("ANOMALY VALIDATION")
    print("=" * 70)

    print("\nTotal transactions:", len(df))
    print("HIGH-risk transactions:", high_risk.sum())
    print(
        "HIGH-risk possible own-account transfers:",
        len(own_account_high_risk)
    )

    print("\nTOP POSSIBLE OWN-ACCOUNT TRANSFERS:\n")

    columns = [
        "id",
        "account_id",
        "value_date",
        "amount",
        "debit",
        "credit",
        "risk_score",
        "risk_level",
        "raw_narration",
    ]

    print(
        own_account_high_risk
        .head(20)[columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()