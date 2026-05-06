"""
Utility functions for working with the AutoGluon-TimeSeries benchmark datasets.
Includes export to various formats (CSV, Parquet, etc.)
"""

import pandas as pd
import json
from pathlib import Path
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.common import ListDataset
import numpy as np

DATASET_NAMES = [
    "car_parts_without_missing",
    "cif_2016",
    "covid_deaths",
    "electricity_hourly",
    "electricity_weekly",
    "fred_md",
    "hospital",
    "kdd_cup_2018_without_missing",
    "m1_monthly",
    "m1_quarterly",
    "m1_yearly",
    "m3_monthly",
    "m3_other",
    "m3_quarterly",
    "m3_yearly",
    "m4_daily",
    "m4_hourly",
    "m4_monthly",
    "m4_quarterly",
    "m4_weekly",
    "m4_yearly",
    "nn5_daily_without_missing",
    "nn5_weekly",
    "pedestrian_counts",
    "tourism_monthly",
    "tourism_quarterly",
    "tourism_yearly",
    "vehicle_trips_without_missing",
    "web_traffic_weekly",
]


def export_dataset_to_csv(dataset_name, output_dir="./exported_datasets", split="train"):
    """
    Export a GluonTS dataset to CSV format (long format).

    Args:
        dataset_name: Name of the dataset
        output_dir: Directory to save CSV files
        split: Either 'train' or 'test'
    """
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Loading {dataset_name} ({split} split)...")
    dataset = get_dataset(dataset_name, regenerate=False)
    data = dataset.train if split == "train" else dataset.test

    # Convert to long format DataFrame
    rows = []
    for i, entry in enumerate(data):
        item_id = entry.get("item_id", f"item_{i}")
        target = entry["target"]
        start = entry["start"]

        # Create timestamps
        timestamps = pd.date_range(
            start=start.to_timestamp(),
            periods=len(target),
            freq=dataset.metadata.freq
        )

        for timestamp, value in zip(timestamps, target):
            rows.append({
                "item_id": item_id,
                "timestamp": timestamp,
                "target": value
            })

    df = pd.DataFrame(rows)
    csv_path = output_path / f"{split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Exported to {csv_path}")

    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "prediction_length": int(dataset.metadata.prediction_length),
        "freq": dataset.metadata.freq,
        "num_time_series": len(list(data)),
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_path}")

    return df


def get_dataset_statistics(dataset_name):
    """Get comprehensive statistics for a dataset."""
    dataset = get_dataset(dataset_name, regenerate=False)

    train_data = list(dataset.train)
    test_data = list(dataset.test)

    stats = {
        "name": dataset_name,
        "num_train_series": len(train_data),
        "num_test_series": len(test_data),
        "prediction_length": int(dataset.metadata.prediction_length),
        "frequency": dataset.metadata.freq,
    }

    # Calculate lengths
    train_lengths = [len(entry["target"]) for entry in train_data]
    stats["min_length"] = min(train_lengths)
    stats["max_length"] = max(train_lengths)
    stats["mean_length"] = np.mean(train_lengths)
    stats["total_timesteps"] = sum(train_lengths)

    return stats


def export_all_datasets(output_dir="./exported_datasets", formats=["csv"]):
    """
    Export all datasets to specified formats.

    Args:
        output_dir: Directory to save exported datasets
        formats: List of formats to export to (currently only 'csv' supported)
    """
    print(f"Exporting {len(DATASET_NAMES)} datasets...")
    print("=" * 80)

    all_stats = []

    for i, dataset_name in enumerate(DATASET_NAMES, 1):
        print(f"\n[{i}/{len(DATASET_NAMES)}] Processing: {dataset_name}")
        try:
            # Export train split
            export_dataset_to_csv(dataset_name, output_dir, split="train")

            # Export test split
            export_dataset_to_csv(dataset_name, output_dir, split="test")

            # Get statistics
            stats = get_dataset_statistics(dataset_name)
            all_stats.append(stats)

        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")

    # Save summary statistics
    stats_df = pd.DataFrame(all_stats)
    stats_path = Path(output_dir) / "dataset_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✓ Saved summary statistics to {stats_path}")

    return stats_df


def load_dataset_as_dataframe(dataset_name, split="train"):
    """
    Load a dataset directly as a pandas DataFrame.

    Args:
        dataset_name: Name of the dataset
        split: Either 'train' or 'test'

    Returns:
        pandas DataFrame in long format
    """
    dataset = get_dataset(dataset_name, regenerate=False)
    data = dataset.train if split == "train" else dataset.test

    rows = []
    for i, entry in enumerate(data):
        item_id = entry.get("item_id", f"item_{i}")
        target = entry["target"]
        start = entry["start"]

        timestamps = pd.date_range(
            start=start.to_timestamp(),
            periods=len(target),
            freq=dataset.metadata.freq
        )

        for timestamp, value in zip(timestamps, target):
            rows.append({
                "item_id": item_id,
                "timestamp": timestamp,
                "target": value
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export AutoGluon-TimeSeries benchmark datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset to export (exports all if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./exported_datasets",
        help="Output directory for exported datasets"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only generate statistics, don't export datasets"
    )

    args = parser.parse_args()

    if args.stats_only:
        print("Generating statistics for all datasets...")
        all_stats = []
        for dataset_name in DATASET_NAMES:
            try:
                stats = get_dataset_statistics(dataset_name)
                all_stats.append(stats)
                print(f"✓ {dataset_name}")
            except Exception as e:
                print(f"✗ {dataset_name}: {e}")

        stats_df = pd.DataFrame(all_stats)
        print("\n" + "=" * 80)
        print(stats_df.to_string())

    elif args.dataset:
        print(f"Exporting dataset: {args.dataset}")
        export_dataset_to_csv(args.dataset, args.output_dir, split="train")
        export_dataset_to_csv(args.dataset, args.output_dir, split="test")
        stats = get_dataset_statistics(args.dataset)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        export_all_datasets(args.output_dir)