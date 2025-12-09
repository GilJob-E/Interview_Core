"""Feature statistics calculation from interview dataset.

Calculates mean, std, min, max, percentiles for 4 Tier 1 features
using Welford's algorithm for streaming computation.

Tier 1 features appear in Top 20 for ALL 5 traits in Naim et al. (2018).
"""

import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Tier 1 features from Naim et al. (2018) Table 6
# These appear in Top 20 for ALL 5 traits
TARGET_FEATURES = [
    "wpsec",           # words per second
    "upsec",           # unique words per second
    "fpsec",           # fillers per second
    "quantifier_ratio",
]


@dataclass
class FeatureStats:
    """Statistics for a single feature."""
    name: str
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    p5: float = 0.0
    p25: float = 0.0
    p50: float = 0.0  # median
    p75: float = 0.0
    p95: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min_val, 6),
            "max": round(self.max_val, 6),
            "percentiles": {
                "p5": round(self.p5, 6),
                "p25": round(self.p25, 6),
                "p50": round(self.p50, 6),
                "p75": round(self.p75, 6),
                "p95": round(self.p95, 6),
            }
        }


class WelfordAccumulator:
    """Online mean/variance calculation using Welford's algorithm.

    Memory-efficient streaming statistics calculation.
    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from mean
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, value: float) -> None:
        """Update statistics with new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def variance(self) -> float:
        """Get sample variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Get sample standard deviation."""
        return math.sqrt(self.variance)


class ReservoirSampler:
    """Reservoir sampling for percentile estimation.

    Maintains a fixed-size sample for memory-efficient percentile calculation.
    """

    def __init__(self, size: int = 10000):
        self.size = size
        self.samples: List[float] = []
        self.count = 0

    def add(self, value: float) -> None:
        """Add value to reservoir."""
        self.count += 1

        if len(self.samples) < self.size:
            self.samples.append(value)
        else:
            # Reservoir sampling: replace with probability size/count
            j = random.randint(0, self.count - 1)
            if j < self.size:
                self.samples[j] = value

    def percentile(self, p: float) -> float:
        """Calculate p-th percentile (0-100)."""
        if not self.samples:
            return 0.0

        sorted_samples = sorted(self.samples)
        k = (len(sorted_samples) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_samples[int(k)]

        return sorted_samples[int(f)] * (c - k) + sorted_samples[int(c)] * (k - f)


class StreamingStatsCalculator:
    """Calculate streaming statistics for multiple features."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.accumulators: Dict[str, WelfordAccumulator] = {
            name: WelfordAccumulator() for name in feature_names
        }
        self.reservoirs: Dict[str, ReservoirSampler] = {
            name: ReservoirSampler(size=10000) for name in feature_names
        }

    def update(self, features: Dict[str, float]) -> None:
        """Update statistics with feature dictionary."""
        for name in self.feature_names:
            if name in features:
                value = features[name]
                if value is not None and not math.isnan(value):
                    self.accumulators[name].update(value)
                    self.reservoirs[name].add(value)

    def get_statistics(self) -> Dict[str, FeatureStats]:
        """Get computed statistics for all features."""
        stats = {}
        for name in self.feature_names:
            acc = self.accumulators[name]
            res = self.reservoirs[name]

            stats[name] = FeatureStats(
                name=name,
                count=acc.count,
                mean=acc.mean,
                std=acc.std,
                min_val=acc.min_val if acc.count > 0 else 0.0,
                max_val=acc.max_val if acc.count > 0 else 0.0,
                p5=res.percentile(5),
                p25=res.percentile(25),
                p50=res.percentile(50),
                p75=res.percentile(75),
                p95=res.percentile(95),
            )

        return stats


def load_json_file(file_path: Path) -> Optional[Dict]:
    """Load and parse a single JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def extract_sample_data(data: Dict) -> Optional[Tuple[str, float]]:
    """Extract text and duration from JSON data.

    Returns:
        Tuple of (text, duration_seconds) or None if invalid.
    """
    try:
        text = data["dataSet"]["answer"]["raw"]["text"]
        duration_ms = data["rawDataInfo"]["answer"]["duration"]

        # Validate
        if not text or len(text.strip()) < 10:
            return None
        if duration_ms <= 0:
            return None

        duration_sec = duration_ms / 1000.0
        return (text, duration_sec)

    except (KeyError, TypeError):
        return None


def process_single_file(
    file_path: str,
    analyzer: "InterviewAnalyzer" = None,
) -> Optional[Dict[str, float]]:
    """Process a single JSON file and extract features.

    Args:
        file_path: Path to JSON file.
        analyzer: Shared InterviewAnalyzer instance (for performance).
    """
    path = Path(file_path)
    data = load_json_file(path)
    if data is None:
        return None

    sample = extract_sample_data(data)
    if sample is None:
        return None

    text, duration = sample

    try:
        # Use provided analyzer or create new one
        if analyzer is None:
            from ko_liwc.analyzer import InterviewAnalyzer
            analyzer = InterviewAnalyzer()

        features = analyzer.extract_features(text, duration)

        # Filter to target features only
        return {k: v for k, v in features.items() if k in TARGET_FEATURES}

    except Exception:
        return None


def discover_json_files(data_dir: Path) -> List[Path]:
    """Discover all JSON files in data directory."""
    return list(data_dir.rglob("*.json"))


def calculate_statistics(
    data_dir: str,
    output_path: Optional[str] = None,
    batch_size: int = 1000,
    max_workers: int = 4,
    verbose: bool = True,
) -> Dict[str, FeatureStats]:
    """Calculate feature statistics from interview dataset.

    Args:
        data_dir: Path to test_data directory.
        output_path: Path to save statistics JSON.
        batch_size: Number of files per batch.
        max_workers: Number of parallel workers.
        verbose: Print progress messages.

    Returns:
        Dictionary of feature name to FeatureStats.
    """
    data_path = Path(data_dir)

    if verbose:
        print(f"Discovering JSON files in {data_path}...")

    json_files = discover_json_files(data_path)
    total_files = len(json_files)

    if verbose:
        print(f"Found {total_files} JSON files")

    # Initialize streaming calculator
    calculator = StreamingStatsCalculator(TARGET_FEATURES)

    # Process files
    processed = 0
    skipped = 0

    # Create shared analyzer instance (avoids repeated Kiwi initialization)
    if verbose:
        print("Initializing analyzer...")

    from ko_liwc.analyzer import InterviewAnalyzer
    analyzer = InterviewAnalyzer()

    if verbose:
        print("Extracting features...")

    # Process in batches for progress reporting
    for i in range(0, total_files, batch_size):
        batch_files = json_files[i:i + batch_size]

        for file_path in batch_files:
            features = process_single_file(str(file_path), analyzer=analyzer)

            if features is not None:
                calculator.update(features)
                processed += 1
            else:
                skipped += 1

        if verbose:
            progress = min(i + batch_size, total_files)
            print(f"Progress: {progress}/{total_files} ({100*progress/total_files:.1f}%)")

    if verbose:
        print(f"\nProcessed: {processed}, Skipped: {skipped}")

    # Get final statistics
    statistics = calculator.get_statistics()

    # Save to JSON if output path specified
    if output_path:
        output = {
            "metadata": {
                "total_files": total_files,
                "processed": processed,
                "skipped": skipped,
                "features": TARGET_FEATURES,
                "data_dir": str(data_path),
                "generated_at": datetime.now().isoformat(),
            },
            "statistics": {
                name: stats.to_dict()
                for name, stats in statistics.items()
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"Statistics saved to {output_path}")

    return statistics


def load_statistics(path: str) -> Dict[str, FeatureStats]:
    """Load statistics from JSON file.

    Args:
        path: Path to statistics JSON file.

    Returns:
        Dictionary of feature name to FeatureStats.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {}
    for name, stat_dict in data["statistics"].items():
        stats[name] = FeatureStats(
            name=stat_dict["name"],
            count=stat_dict["count"],
            mean=stat_dict["mean"],
            std=stat_dict["std"],
            min_val=stat_dict["min"],
            max_val=stat_dict["max"],
            p5=stat_dict["percentiles"]["p5"],
            p25=stat_dict["percentiles"]["p25"],
            p50=stat_dict["percentiles"]["p50"],
            p75=stat_dict["percentiles"]["p75"],
            p95=stat_dict["percentiles"]["p95"],
        )

    return stats


def print_statistics_summary(statistics: Dict[str, FeatureStats]) -> None:
    """Print formatted statistics summary."""
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Feature':<20} {'Count':>8} {'Mean':>10} {'Std':>10} "
          f"{'Min':>10} {'Max':>10} {'P50':>10}")
    print("-" * 80)

    for name in TARGET_FEATURES:
        if name in statistics:
            s = statistics[name]
            print(f"{s.name:<20} {s.count:>8} {s.mean:>10.4f} {s.std:>10.4f} "
                  f"{s.min_val:>10.4f} {s.max_val:>10.4f} {s.p50:>10.4f}")

    print("=" * 80)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate feature statistics from interview dataset"
    )
    parser.add_argument(
        "--data-dir",
        default="test_data",
        help="Path to test_data directory"
    )
    parser.add_argument(
        "--output",
        default="ko_liwc/data/feature_statistics.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for progress reporting"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )

    args = parser.parse_args()

    # Run calculation
    stats = calculate_statistics(
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Print summary
    print_statistics_summary(stats)
