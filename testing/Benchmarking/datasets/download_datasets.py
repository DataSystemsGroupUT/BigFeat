"""
Download all 29 benchmark datasets used in AutoGluon-TimeSeries paper.
These datasets are available through GluonTS.
"""

import os
from pathlib import Path
from gluonts.dataset.repository import get_dataset

# List of all 29 datasets from Table 8 in the paper
DATASETS = [
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
    "kaggle_web_traffic_weekly",  # Corrected name for web_traffic_weekly
]

def download_all_datasets(output_dir="./datasets"):
    """Download all benchmark datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Downloading {len(DATASETS)} datasets to {output_path.absolute()}")
    print("=" * 80)

    # Check for M3 datasets special case
    m3_datasets = [d for d in DATASETS if d.startswith("m3_")]
    if m3_datasets:
        print("\nNOTE: M3 datasets require manual download from:")
        print("https://forecasters.org/resources/time-series-data/m3-competition/")
        print("Download M3C.xls and place it in: ~/.gluonts/datasets/M3C.xls")
        print("=" * 80)

    successful = []
    failed = []

    for i, dataset_name in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] Downloading: {dataset_name}")
        try:
            dataset = get_dataset(dataset_name, regenerate=False)
            print(f"  ✓ Successfully downloaded {dataset_name}")
            print(f"    - Training samples: {len(list(dataset.train))}")
            print(f"    - Test samples: {len(list(dataset.test))}")
            print(f"    - Prediction length: {dataset.metadata.prediction_length}")
            print(f"    - Frequency: {dataset.metadata.freq}")
            successful.append(dataset_name)
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Failed to download {dataset_name}: {error_msg}")

            # Provide helpful instructions for common errors
            if "m3" in dataset_name.lower() and "M3C.xls" in error_msg:
                print(f"     → M3 datasets require manual download")
                print(f"     → Visit: https://forecasters.org/resources/time-series-data/m3-competition/")
                print(f"     → Download M3C.xls to: ~/.gluonts/datasets/M3C.xls")

            failed.append((dataset_name, error_msg))

    print("\n" + "=" * 80)
    print(f"\nSummary:")
    print(f"  Successfully downloaded: {len(successful)}/{len(DATASETS)}")
    print(f"  Failed: {len(failed)}/{len(DATASETS)}")

    if failed:
        print(f"\nFailed datasets:")
        for name, error in failed:
            if "m3" in name.lower():
                print(f"  - {name}: Requires manual download (see instructions above)")
            else:
                print(f"  - {name}: {error[:100]}...")

    print(f"\nDatasets are cached in: ~/gluonts/datasets/")
    print(f"You can access them using: get_dataset('dataset_name')")

if __name__ == "__main__":
    print("AutoGluon-TimeSeries Benchmark Dataset Downloader")
    print("=" * 80)
    print("\nThis script will download all 29 datasets used in the paper:")
    print("'AutoGluon-TimeSeries: AutoML for Probabilistic Time Series Forecasting'")
    print("\nNote: Datasets are cached by GluonTS in ~/gluonts/datasets/")
    print("Total size may be several GB depending on the datasets.\n")

    response = input("Continue with download? (y/n): ")
    if response.lower() == 'y':
        download_all_datasets()
    else:
        print("Download cancelled.")