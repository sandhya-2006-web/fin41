import json
from pathlib import Path
from collections import Counter


DATA_PATH = Path("data/raw/bank_transactions.json")


def main():
    print("=" * 70)
    print("FIN-41 DATASET PROFILE")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nPython type: {type(data).__name__}")

    # ---------------------------------------------------------
    # Determine records
    # ---------------------------------------------------------
    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):
        print("\nTop-level keys:")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"  {key}: list ({len(value)} records)")

        list_values = [
            value for value in data.values()
            if isinstance(value, list)
        ]

        if not list_values:
            print("\nDataset is a dictionary but contains no list of records.")
            print("Top-level structure:")
            print(data)
            return

        records = max(list_values, key=len)

    else:
        print("Unsupported dataset structure.")
        return

    print(f"\nNumber of records: {len(records)}")

    if not records:
        print("Dataset contains zero records.")
        return

    # ---------------------------------------------------------
    # Fields
    # ---------------------------------------------------------
    if not isinstance(records[0], dict):
        print("\nFirst record:")
        print(records[0])
        return

    fields = list(records[0].keys())

    print(f"\nNumber of fields: {len(fields)}")
    print("\nFields:")
    for field in fields:
        print(f"  - {field}")

    # ---------------------------------------------------------
    # Field types
    # ---------------------------------------------------------
    print("\nField types:")
    for field in fields:
        values = [
            record.get(field)
            for record in records[:100]
            if isinstance(record, dict)
            and record.get(field) is not None
        ]

        types = Counter(type(v).__name__ for v in values)

        print(f"  {field}: {dict(types)}")

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------
    print("\nMissing values:")
    for field in fields:
        missing = sum(
            1
            for record in records
            if record.get(field) is None
        )

        percentage = missing / len(records) * 100

        print(
            f"  {field}: "
            f"{missing}/{len(records)} "
            f"({percentage:.2f}%)"
        )

    # ---------------------------------------------------------
    # Sample records
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("FIRST 3 RECORDS")
    print("=" * 70)

    for i, record in enumerate(records[:3], start=1):
        print(f"\nRecord {i}:")
        print(json.dumps(record, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------
    # Duplicate records
    # ---------------------------------------------------------
    serialized = [
        json.dumps(record, sort_keys=True, default=str)
        for record in records
    ]

    duplicates = len(serialized) - len(set(serialized))

    print("\n" + "=" * 70)
    print("DUPLICATES")
    print("=" * 70)

    print(f"Duplicate complete records: {duplicates}")


if __name__ == "__main__":
    main()