"""Command-line entry point for Lacuna Step 6 temporal backtesting."""

from __future__ import annotations

from pathlib import Path

from lacuna.models.backtest import load_dataset, run_temporal_backtest
from lacuna.models.io import save_model_artifact
from lacuna.models.train import FEATURE_COLUMNS


DATA_DIR = Path("data/processed/ml")
MODEL_DIR = Path("artifacts/models")


def load_all_datasets():
    """Load all labelled historical ML datasets."""
    return {
        year: load_dataset(
            DATA_DIR / f"lacuna_ml_dataset_{year}.pkl"
        )
        for year in (2010, 2015, 2020)
    }


def print_result(result: dict) -> None:
    """Print one backtest result in a compact research-friendly format."""
    metrics = result["metrics"]

    print(
        f"{result['model_name']:<24}"
        f"PR-AUC={metrics['pr_auc']:.4f}  "
        f"ROC-AUC={metrics['roc_auc']:.4f}  "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"P@10={metrics['precision_at_10']:.4f}  "
        f"P@50={metrics['precision_at_50']:.4f}  "
        f"P@100={metrics['precision_at_100']:.4f}"
    )


def main() -> None:
    datasets = load_all_datasets()

    backtests = [
        ((2010,), 2015),
        ((2010, 2015), 2020),
    ]

    all_results = []

    print("\nLacuna Temporal Backtesting")
    print("=" * 110)

    for train_cutoffs, test_cutoff in backtests:
        print(
            f"\nTrain cutoffs: {train_cutoffs} "
            f"→ Test cutoff: {test_cutoff}"
        )
        print("-" * 110)

        results = run_temporal_backtest(
            datasets=datasets,
            train_cutoffs=train_cutoffs,
            test_cutoff=test_cutoff,
        )

        for result in results:
            print_result(result)
            all_results.append(result)

            artifact_name = (
                f"{result['model_name']}"
                f"_train_{'_'.join(map(str, train_cutoffs))}"
                f"_test_{test_cutoff}.joblib"
            )

            metadata = {
                "model_name": result["model_name"],
                "train_cutoffs": result["train_cutoffs"],
                "test_cutoff": result["test_cutoff"],
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                "train_positive_rate": result["train_positive_rate"],
                "test_positive_rate": result["test_positive_rate"],
                "feature_columns": FEATURE_COLUMNS,
                "metrics": result["metrics"],
            }

            save_model_artifact(
                result["model"],
                MODEL_DIR / artifact_name,
                metadata,
            )

    print("\n" + "=" * 110)
    print(
        f"Completed {len(all_results)} model backtests."
    )
    print(
        f"Artifacts saved to: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()
