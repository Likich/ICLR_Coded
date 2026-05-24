#!/usr/bin/env python3
"""Validate the public ICLR review-code release files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FORBIDDEN_COLUMN_PATTERNS = [
    "raw_text",
    "review_text",
    "strengths_text",
    "weaknesses_text",
    "text",
    "title",
    "author",
    "reviewer",
    "email",
]

EXPECTED_SEGMENTS = 39_979
EXPECTED_REVIEWS = 30_000
EXPECTED_SEGMENT_COUNTS = {
    "full_review": 20_021,
    "strength": 9_979,
    "weakness": 9_979,
}
VALID_YEARS = set(range(2020, 2026))
VALID_SEGMENT_TYPES = set(EXPECTED_SEGMENT_COUNTS)


def require_columns(dataframe: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise AssertionError(f"{source} is missing required columns: {missing}")


def check_forbidden_columns(dataframe: pd.DataFrame, source: str) -> None:
    forbidden = []
    for column in dataframe.columns:
        lower_column = column.lower()
        if any(pattern in lower_column for pattern in FORBIDDEN_COLUMN_PATTERNS):
            forbidden.append(column)
    if forbidden:
        raise AssertionError(f"{source} contains forbidden public-release columns: {forbidden}")


def check_non_empty(dataframe: pd.DataFrame, column: str, source: str) -> None:
    values = dataframe[column]
    missing = values.isna() | (values.astype(str).str.strip() == "")
    if missing.any():
        raise AssertionError(f"{source}.{column} has {int(missing.sum())} empty values")


def validate_main_release(path: Path) -> pd.DataFrame:
    release = pd.read_csv(path)
    require_columns(
        release,
        [
            "year",
            "paper_id",
            "review_id",
            "segment_id",
            "segment_type",
            "segment_hash",
            "canonical_code",
            "axial_cluster_id",
        ],
        str(path),
    )
    check_forbidden_columns(release, str(path))

    if len(release) != EXPECTED_SEGMENTS:
        raise AssertionError(f"Expected {EXPECTED_SEGMENTS} segments, found {len(release)}")
    review_count = release["review_id"].nunique()
    if review_count != EXPECTED_REVIEWS:
        raise AssertionError(f"Expected {EXPECTED_REVIEWS} reviews, found {review_count}")

    segment_counts = release["segment_type"].value_counts().to_dict()
    for segment_type, expected_count in EXPECTED_SEGMENT_COUNTS.items():
        observed_count = int(segment_counts.get(segment_type, 0))
        if observed_count != expected_count:
            raise AssertionError(
                f"Expected {expected_count} {segment_type} segments, found {observed_count}"
            )

    if release["segment_id"].duplicated().any():
        duplicate_count = int(release["segment_id"].duplicated().sum())
        raise AssertionError(f"segment_id has {duplicate_count} duplicates")

    check_non_empty(release, "segment_hash", str(path))
    check_non_empty(release, "canonical_code", str(path))
    check_non_empty(release, "axial_cluster_id", str(path))

    years = set(release["year"].astype(int).unique())
    if years - VALID_YEARS:
        raise AssertionError(f"Invalid years found: {sorted(years - VALID_YEARS)}")
    segment_types = set(release["segment_type"].astype(str).unique())
    if segment_types - VALID_SEGMENT_TYPES:
        raise AssertionError(f"Invalid segment types found: {sorted(segment_types - VALID_SEGMENT_TYPES)}")

    return release


def validate_auxiliary_file(path: Path, required_columns: list[str]) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    require_columns(dataframe, required_columns, str(path))
    check_forbidden_columns(dataframe, str(path))
    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate public release files.")
    parser.add_argument("--release-dir", type=Path, default=Path("data/release"))
    args = parser.parse_args()

    release_dir = args.release_dir
    main_release = validate_main_release(release_dir / "iclr_review_codes_2020_2025.csv")
    manual = validate_auxiliary_file(
        release_dir / "manual_reference_codes.csv",
        [
            "year",
            "paper_id",
            "review_id",
            "segment_id",
            "segment_type",
            "segment_hash",
            "human_open_code",
            "raw_open_code",
            "canonical_code",
            "rouge_l_to_human",
            "minilm_cosine_to_human",
        ],
    )
    clusters = validate_auxiliary_file(
        release_dir / "cluster_metadata.csv",
        [
            "axial_cluster_id",
            "axial_cluster_label",
            "n_segments",
            "n_reviews",
            "n_papers",
            "mean_review_score",
            "acceptance_rate",
            "top_canonical_codes",
            "representative_segment_hashes",
        ],
    )
    codebook = validate_auxiliary_file(
        release_dir / "codebook.csv",
        [
            "canonical_code_id",
            "canonical_code",
            "raw_code_variants",
            "count_segments",
            "count_reviews",
            "count_papers",
            "years_observed",
            "strength_count",
            "weakness_count",
            "full_review_count",
            "mean_review_score",
            "acceptance_rate",
        ],
    )

    print("Release validation passed")
    print(f"segments={len(main_release):,}")
    print(f"reviews={main_release['review_id'].nunique():,}")
    print(f"manual_reference_segments={len(manual):,}")
    print(f"clusters={len(clusters):,}")
    print(f"canonical_codes={len(codebook):,}")


if __name__ == "__main__":
    main()
