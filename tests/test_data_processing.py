import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data_processing import (  # noqa: E402
    simple_process,
    create_customer_aggregates,
)


def test_simple_process_adds_datetime_parts():
    data = {
        "TransactionId": ["t1", "t2"],
        "CustomerId": ["c1", "c2"],
        "TransactionStartTime": [
            "2019-01-01T10:00:00Z",
            "2019-01-02T15:30:00Z",
        ],
        "Amount": [100.0, 200.0],
    }
    df = pd.DataFrame(data)

    df_proc = simple_process(df)

    for col in [
        "TransactionYear",
        "TransactionMonth",
        "TransactionDay",
        "TransactionHour",
    ]:
        assert col in df_proc.columns


def test_create_customer_aggregates_returns_expected_columns():
    data = {
        "CustomerId": ["c1", "c1", "c2"],
        "Amount": [100.0, 200.0, 50.0],
    }
    df = pd.DataFrame(data)

    agg_df = create_customer_aggregates(df)

    expected_cols = {
        "CustomerId",
        "total_amount",
        "avg_amount",
        "std_amount",
        "tx_count",
    }
    assert expected_cols.issubset(set(agg_df.columns))

    row_c1 = agg_df[agg_df["CustomerId"] == "c1"].iloc[0]
    assert row_c1["total_amount"] == 300.0
    assert row_c1["tx_count"] == 2
