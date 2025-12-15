from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(file_name: str = "data.csv") -> pd.DataFrame:
    """Load raw data from data/raw."""
    raw_path = Path(__file__).resolve().parents[1] / "data" / "raw" / file_name
    print("Loading raw data from:", raw_path)
    df = pd.read_csv(raw_path)
    return df


def simple_process(df: pd.DataFrame) -> pd.DataFrame:
    """Basic processing with datetime features."""
    df = df.copy()

    if "TransactionStartTime" in df.columns:
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"],
            errors="coerce",
        )
        df["TransactionYear"] = df["TransactionStartTime"].dt.year
        df["TransactionMonth"] = df["TransactionStartTime"].dt.month
        df["TransactionDay"] = df["TransactionStartTime"].dt.day
        df["TransactionHour"] = df["TransactionStartTime"].dt.hour

    return df


def create_customer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-customer aggregate features:
    total amount, average amount, transaction count, std of amount.
    """
    df = df.copy()

    if "CustomerId" not in df.columns:
        raise ValueError("CustomerId column is missing from dataframe")

    if "Amount" not in df.columns:
        raise ValueError("Amount column is missing from dataframe")

    grouped = df.groupby("CustomerId")

    agg_df = grouped["Amount"].agg(
        total_amount="sum",
        avg_amount="mean",
        std_amount="std",
    )

    agg_df["tx_count"] = grouped.size()
    agg_df = agg_df.reset_index()

    return agg_df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary (RFM) per CustomerId.
    Recency is in days since last transaction.
    """
    df = df.copy()

    if "CustomerId" not in df.columns:
        raise ValueError("CustomerId column is missing")
    if "TransactionStartTime" not in df.columns:
        raise ValueError("TransactionStartTime column is missing")
    if "Amount" not in df.columns and "Value" not in df.columns:
        raise ValueError("Neither Amount nor Value column is present")

    if "Amount" in df.columns:
        df["Monetary"] = df["Amount"]
    else:
        df["Monetary"] = df["Value"]

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"],
        errors="coerce",
    )

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (snapshot_date - x.max()).days,
        ),
        Frequency=("TransactionId", "count"),
        Monetary=("Monetary", "sum"),
    ).reset_index()

    return rfm


def label_high_risk_from_rfm(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster customers based on RFM and create is_high_risk label.
    High-risk cluster = low Frequency and low Monetary.
    """
    rfm = rfm.copy()

    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    X_rfm = scaler.fit_transform(rfm[features])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    rfm["cluster"] = kmeans.fit_predict(X_rfm)

    cluster_stats = rfm.groupby("cluster")[features].mean()

    high_risk_cluster = cluster_stats.sort_values(
        by=["Frequency", "Monetary"],
        ascending=[True, True],
    ).index[0]

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[
        ["CustomerId", "Recency", "Frequency", "Monetary", "is_high_risk"]
    ]


def build_preprocessing_pipeline(df: pd.DataFrame) -> Pipeline:
    """
    Build a preprocessing pipeline:
    impute missing values, scale numeric, one-hot encode categorical.
    """
    numeric_features = df.select_dtypes(
        include=["int64", "float64"],
    ).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object"],
    ).columns.tolist()

    id_cols = [
        "CustomerId",
        "AccountId",
        "TransactionId",
        "BatchId",
        "SubscriptionId",
    ]
    numeric_features = [c for c in numeric_features if c not in id_cols]
    categorical_features = [
        c
        for c in categorical_features
        if c not in id_cols
                            ]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
    )

    return preprocessor


def save_processed_data(
    df: pd.DataFrame,
    file_name: str = "processed_simple.csv",
) -> None:
    """Save processed data to data/processed."""
    processed_path = (
        Path(__file__).resolve().parents[1] / "data" / "processed" / file_name
    )
    print("Saving processed data to:", processed_path)
    df.to_csv(processed_path, index=False)


def main() -> None:
    df_raw = load_raw_data("data.csv")

    df_proc = simple_process(df_raw)

    df_customer = create_customer_aggregates(df_proc)

    rfm = compute_rfm(df_proc)
    rfm_with_label = label_high_risk_from_rfm(rfm)

    df_train = df_customer.merge(
        rfm_with_label,
        on="CustomerId",
        how="left",
    )

    save_processed_data(
        df_train,
        file_name="train_customers_with_target.csv",
    )

    preprocessor = build_preprocessing_pipeline(df_train)
    X = preprocessor.fit_transform(df_train)

    print("Training data shape:", df_train.shape)
    print(
        "Preprocessed feature matrix shape:",
        X.shape,
    )


if __name__ == "__main__":
    main()
