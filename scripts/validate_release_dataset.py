#!/usr/bin/env python3
"""Validate the public open-code demonstration sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_SEGMENTS = 1_000
EXPECTED_CODES = 628
VALID_YEARS = set(range(2020, 2026))
VALID_SEGMENT_TYPES = {"full_review", "strength", "weakness"}
SAMPLE_COLUMNS = [
    "year",
    "venue",
    "paper_id",
    "review_id",
    "segment_id",
    "segment_type",
    "segment_hash",
    "paper_decision",
    "review_score",
    "review_confidence",
    "raw_open_code",
    "canonical_code",
    "canonical_code_id",
    "split",
    "source_available_from",
    "code_model",
    "moderator_model",
]
CODEBOOK_COLUMNS = [
    "canonical_code_id",
    "canonical_code",
    "raw_code_variants",
    "count_segments_in_sample",
    "count_reviews_in_sample",
    "count_papers_in_sample",
    "years_observed_in_sample",
    "strength_count_in_sample",
    "weakness_count_in_sample",
    "full_review_count_in_sample",
    "mean_review_score_in_sample",
    "acceptance_rate_in_sample",
]
WITHHELD_FILES = [
    "cluster_metadata.csv",
    "iclr_review_codes_2020_2025.csv",
    "manual_reference_codes.csv",
    "codebook.csv",
    "release_stats.json",
]


def require_exact_columns(dataframe: pd.DataFrame, expected: list[str], source: Path) -> None:
    if dataframe.columns.tolist() != expected:
        raise AssertionError(
            f"{source} has unexpected schema.\n"
            f"Expected: {expected}\nObserved: {dataframe.columns.tolist()}"
        )


def require_non_empty(dataframe: pd.DataFrame, column: str, source: Path) -> None:
    empty = dataframe[column].isna() | (dataframe[column].astype(str).str.strip() == "")
    if empty.any():
        raise AssertionError(f"{source}.{column} has {int(empty.sum())} empty values")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the open-code sample release.")
    parser.add_argument("--release-dir", type=Path, default=Path("data/release"))
    args = parser.parse_args()
    release_dir = args.release_dir

    sample_path = release_dir / "sample_review_codes.csv"
    codebook_path = release_dir / "sample_codebook.csv"
    stats_path = release_dir / "sample_stats.json"
    sample = pd.read_csv(sample_path)
    codebook = pd.read_csv(codebook_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    require_exact_columns(sample, SAMPLE_COLUMNS, sample_path)
    require_exact_columns(codebook, CODEBOOK_COLUMNS, codebook_path)
    if len(sample) != EXPECTED_SEGMENTS:
        raise AssertionError(f"Expected {EXPECTED_SEGMENTS} segment rows, found {len(sample)}")
    if len(codebook) != EXPECTED_CODES:
        raise AssertionError(f"Expected {EXPECTED_CODES} codebook rows, found {len(codebook)}")
    if sample["segment_id"].duplicated().any():
        raise AssertionError("sample_review_codes.csv contains duplicate segment IDs")
    if set(sample["year"].astype(int)) - VALID_YEARS:
        raise AssertionError("sample_review_codes.csv contains years outside 2020-2025")
    if set(sample["segment_type"]) - VALID_SEGMENT_TYPES:
        raise AssertionError("sample_review_codes.csv contains invalid segment types")
    for column in ["segment_hash", "canonical_code", "canonical_code_id"]:
        require_non_empty(sample, column, sample_path)
    if not sample["canonical_code"].str.match(r"^[A-Z]", na=False).all():
        raise AssertionError("All canonical codes must start with a capital letter")
    if set(sample["canonical_code_id"]) != set(codebook["canonical_code_id"]):
        raise AssertionError("Sample and codebook contain inconsistent canonical code IDs")
    if int(codebook["count_segments_in_sample"].sum()) != len(sample):
        raise AssertionError("Codebook frequencies do not sum to sample row count")
    if stats.get("axial_codes_included") is not False:
        raise AssertionError("sample_stats.json must state that axial codes are withheld")
    for name in WITHHELD_FILES:
        if (release_dir / name).exists():
            raise AssertionError(f"Withheld release artifact is present: {name}")

    print("Open-code sample validation passed")
    print(f"records={len(sample):,}")
    print(f"reviews={sample['review_id'].nunique():,}")
    print(f"papers={sample['paper_id'].nunique():,}")
    print(f"canonical_codes={len(codebook):,}")


if __name__ == "__main__":
    main()
