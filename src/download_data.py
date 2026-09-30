import json
import os

from nova_client import list_all


def main():
    print("Downloading bank transactions...")

    transactions = list_all(
        "/bank-transactions",
        sort="value_date",
        order="asc"
    )

    print(f"Downloaded {len(transactions)} transactions")

    os.makedirs("data/raw", exist_ok=True)

    output_path = "data/raw/bank_transactions.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            transactions,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()