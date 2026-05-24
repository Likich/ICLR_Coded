#!/usr/bin/env python3
"""Build the public derived ICLR review-code release files.

The default release intentionally excludes raw review text. Segment hashes are
computed from normalized internal text so users can verify reconstruction
without redistributing the text itself.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


RELEASE_COLUMNS = [
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
    "axial_cluster_id",
    "axial_cluster_label",
    "meta_cluster_id",
    "meta_cluster_label",
    "paper_safe_entities",
    "split",
    "source_available_from",
    "code_model",
    "moderator_model",
    "embedding_model",
    "clustering_config_id",
]

MANUAL_COLUMNS = [
    "year",
    "paper_id",
    "review_id",
    "segment_id",
    "segment_type",
    "segment_hash",
    "human_open_code",
    "human_theme",
    "raw_open_code",
    "canonical_code",
    "rouge_l_to_human",
    "minilm_cosine_to_human",
]

CLUSTER_COLUMNS = [
    "axial_cluster_id",
    "axial_cluster_label",
    "meta_cluster_id",
    "meta_cluster_label",
    "n_segments",
    "n_reviews",
    "n_papers",
    "mean_review_score",
    "acceptance_rate",
    "top_canonical_codes",
    "representative_segment_hashes",
    "representative_snippet",
]

CODEBOOK_COLUMNS = [
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
]

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

SEGMENT_TYPE_MAP = {
    "full_review": "full_review",
    "strength": "strength",
    "strengths": "strength",
    "weakness": "weakness",
    "weaknesses": "weakness",
}

CODE_MODEL = "Qwen2.5-7B-Instruct; Mistral-7B-Instruct-v0.2; Zephyr-7B-beta"
MODERATOR_MODEL = "Mixtral-8x7B-Instruct-v0.1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2; all-mpnet-base-v2"
CLUSTERING_CONFIG_ID = "mpnet_mcs50_ms10_umap10nn10_md0_0"


def normalize_segment_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    normalized = str(value).replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_short_hash(value: str, length: int = 16) -> str:
    return sha256_text(value)[:length]


def split_review_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    review_key = str(value)
    if "||" in review_key:
        return review_key.split("||", 1)[1]
    return review_key


def normalize_segment_type(value: Any) -> str:
    key = str(value).strip().lower()
    return SEGMENT_TYPE_MAP.get(key, key)


def make_segment_id(review_key: Any, segment_type: Any) -> str:
    identifier = f"{split_review_key(review_key)}||{normalize_segment_type(segment_type)}"
    return f"seg_{stable_short_hash(identifier)}"


def parse_listish(value: Any) -> list[Any]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return list(parsed.items())
        except Exception:
            pass
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    return [text]


def parse_jsonish(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return text


def normalize_entity_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def format_paper_safe_entities(value: Any, safe_entities: dict[str, str]) -> str:
    entities = []
    seen = set()
    for entity in parse_listish(value):
        entity_text = str(entity).strip()
        safe_entity = safe_entities.get(normalize_entity_key(entity_text))
        if safe_entity and safe_entity not in seen:
            entities.append(safe_entity)
            seen.add(safe_entity)
    return "; ".join(entities)


def is_accept_value(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    decision = str(value).lower()
    if "accept" in decision and "reject" not in decision:
        return 1.0
    if "reject" in decision:
        return 0.0
    return float("nan")


def require_columns(dataframe: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def check_forbidden_columns(dataframe: pd.DataFrame, include_raw_text: bool, label: str) -> None:
    if include_raw_text:
        return
    forbidden = []
    for column in dataframe.columns:
        lower_column = column.lower()
        if any(pattern in lower_column for pattern in FORBIDDEN_COLUMN_PATTERNS):
            forbidden.append(column)
    if forbidden:
        raise ValueError(
            f"{label} contains forbidden public-release columns {forbidden}. "
            "Use --include_raw_text only for a deliberate non-public export."
        )


def safe_to_csv(dataframe: pd.DataFrame, path: Path, include_raw_text: bool) -> None:
    check_forbidden_columns(dataframe, include_raw_text, path.name)
    dataframe.to_csv(path, index=False)


def load_meta_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["cluster_id", "display_label", "meta_cluster", "meta_label"]
        )
    meta_map = pd.read_csv(path)
    require_columns(
        meta_map,
        ["cluster_id", "display_label", "meta_cluster", "meta_label"],
        str(path),
    )
    return meta_map


def add_semantic_merge(coded: pd.DataFrame, semmerge_path: Path) -> pd.DataFrame:
    if not semmerge_path.exists():
        coded["canonical_code"] = coded["Moderated_Code_sbert80"].fillna(
            coded["Moderated_Code"]
        )
        return coded
    semmerge = pd.read_csv(semmerge_path)
    require_columns(semmerge, ["code_merged"], str(semmerge_path))
    if len(semmerge) != len(coded):
        raise ValueError(
            f"Semantic merge rows ({len(semmerge)}) do not match coded rows ({len(coded)})."
        )
    coded = coded.copy()
    coded["canonical_code"] = semmerge["code_merged"].values
    return coded


def canonical_code_ids(codes: pd.Series) -> dict[str, str]:
    unique_codes = sorted(
        code for code in codes.fillna("uncoded").astype(str).unique() if code.strip()
    )
    return {code: f"CC{index:06d}" for index, code in enumerate(unique_codes, start=1)}


def build_release_dataframe(
    coded_path: Path,
    semmerge_path: Path,
    meta_map_path: Path,
    paper_safe_entity_stats_path: Path,
    human_anchor_path: Path,
    include_raw_text: bool,
) -> pd.DataFrame:
    coded = pd.read_csv(coded_path)
    require_columns(
        coded,
        [
            "year",
            "paper_id",
            "reviewer_id",
            "decision",
            "score",
            "section",
            "utterance",
            "Moderated_Code",
            "review_key",
            "entities_canonical",
            f"hdbscan_cluster_{CLUSTERING_CONFIG_ID}",
            f"hdbscan_cluster_name_{CLUSTERING_CONFIG_ID}",
            f"hdbscan_label_{CLUSTERING_CONFIG_ID}",
        ],
        str(coded_path),
    )
    coded = add_semantic_merge(coded, semmerge_path)
    coded["canonical_code"] = (
        coded["canonical_code"]
        .fillna(coded["Moderated_Code_sbert80"])
        .fillna(coded["Moderated_Code"])
        .fillna("uncoded")
        .astype(str)
        .replace({"": "uncoded"})
    )
    code_ids = canonical_code_ids(coded["canonical_code"])

    meta_map = load_meta_map(meta_map_path)
    meta_lookup = meta_map.set_index("cluster_id").to_dict(orient="index")
    safe_entities = {}
    if paper_safe_entity_stats_path.exists():
        entity_stats = pd.read_csv(paper_safe_entity_stats_path)
        require_columns(entity_stats, ["entity"], str(paper_safe_entity_stats_path))
        safe_entities = {
            normalize_entity_key(entity): str(entity)
            for entity in entity_stats["entity"].dropna().astype(str)
        }

    manual_segment_ids = set()
    if human_anchor_path.exists():
        human_anchor = pd.read_csv(human_anchor_path)
        if {"review_key", "section"}.issubset(human_anchor.columns):
            manual_segment_ids = {
                make_segment_id(row.review_key, row.section)
                for row in human_anchor[["review_key", "section"]].itertuples(index=False)
            }

    rows = []
    cluster_column = f"hdbscan_cluster_{CLUSTERING_CONFIG_ID}"
    cluster_name_column = f"hdbscan_cluster_name_{CLUSTERING_CONFIG_ID}"
    cluster_label_column = f"hdbscan_label_{CLUSTERING_CONFIG_ID}"
    for internal_row in coded.itertuples(index=False):
        row_data = internal_row._asdict()
        segment_type = normalize_segment_type(row_data["section"])
        segment_id = make_segment_id(row_data["review_key"], segment_type)
        segment_text = normalize_segment_text(row_data["utterance"])
        cluster_id = row_data.get(cluster_column, "")
        try:
            cluster_id_int = int(cluster_id)
        except Exception:
            cluster_id_int = -1
        meta_info = meta_lookup.get(cluster_id_int, {})
        raw_cluster_label = row_data.get(cluster_label_column)
        raw_cluster_name = row_data.get(cluster_name_column)
        axial_label = meta_info.get("display_label") or raw_cluster_label or raw_cluster_name
        if pd.isna(axial_label) or not str(axial_label).strip():
            axial_label = "Other" if cluster_id_int == -1 else f"cluster_{cluster_id_int}"
        meta_cluster_id = meta_info.get("meta_cluster", -1 if cluster_id_int == -1 else "")
        meta_cluster_label = meta_info.get("meta_label", "Other" if cluster_id_int == -1 else "")
        canonical_code = str(row_data["canonical_code"]).strip() or "uncoded"

        release_row = {
            "year": int(row_data["year"]),
            "venue": "ICLR.cc",
            "paper_id": row_data["paper_id"],
            "review_id": split_review_key(row_data["review_key"]),
            "segment_id": segment_id,
            "segment_type": segment_type,
            "segment_hash": sha256_text(segment_text),
            "paper_decision": row_data["decision"],
            "review_score": row_data["score"],
            "review_confidence": "",
            "raw_open_code": "" if pd.isna(row_data["Moderated_Code"]) else row_data["Moderated_Code"],
            "canonical_code": canonical_code,
            "canonical_code_id": code_ids[canonical_code],
            "axial_cluster_id": cluster_id_int,
            "axial_cluster_label": str(axial_label),
            "meta_cluster_id": meta_cluster_id,
            "meta_cluster_label": meta_cluster_label,
            "paper_safe_entities": format_paper_safe_entities(
                row_data["entities_canonical"], safe_entities
            ),
            "split": "full_corpus;manual_reference"
            if segment_id in manual_segment_ids
            else "full_corpus",
            "source_available_from": "OpenReview",
            "code_model": CODE_MODEL,
            "moderator_model": MODERATOR_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "clustering_config_id": CLUSTERING_CONFIG_ID,
        }
        if include_raw_text:
            release_row["segment_text"] = segment_text
        rows.append(release_row)

    release = pd.DataFrame(rows)
    ordered_columns = RELEASE_COLUMNS + (["segment_text"] if include_raw_text else [])
    return release[ordered_columns]


def build_manual_reference(human_anchor_path: Path, include_raw_text: bool) -> pd.DataFrame:
    human_anchor = pd.read_csv(human_anchor_path)
    require_columns(
        human_anchor,
        [
            "year",
            "paper_id",
            "review_key",
            "section",
            "utterance",
            "human_open_code",
            "model_open_code",
            "model_canonical_code",
            "rougeL_canonical",
            "semantic_cosine_canonical",
        ],
        str(human_anchor_path),
    )
    rows = []
    for internal_row in human_anchor.itertuples(index=False):
        row_data = internal_row._asdict()
        segment_type = normalize_segment_type(row_data["section"])
        segment_text = normalize_segment_text(row_data["utterance"])
        release_row = {
            "year": int(row_data["year"]),
            "paper_id": row_data["paper_id"],
            "review_id": split_review_key(row_data["review_key"]),
            "segment_id": make_segment_id(row_data["review_key"], segment_type),
            "segment_type": segment_type,
            "segment_hash": sha256_text(segment_text),
            "human_open_code": row_data["human_open_code"],
            "human_theme": row_data.get("human_theme_optional", ""),
            "raw_open_code": row_data["model_open_code"],
            "canonical_code": row_data["model_canonical_code"],
            "rouge_l_to_human": row_data["rougeL_canonical"],
            "minilm_cosine_to_human": row_data["semantic_cosine_canonical"],
        }
        if include_raw_text:
            release_row["segment_text"] = segment_text
        rows.append(release_row)
    ordered_columns = MANUAL_COLUMNS + (["segment_text"] if include_raw_text else [])
    return pd.DataFrame(rows)[ordered_columns]


def format_top_codes(value: Any) -> str:
    parsed = parse_jsonish(value)
    if parsed is None:
        return ""
    formatted = []
    if isinstance(parsed, dict):
        iterator = parsed.items()
        for code, count in iterator:
            formatted.append(f"{code} ({count})")
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                code = item.get("code") or item.get("label") or item.get("name")
                count = (
                    item.get("count")
                    or item.get("code_count")
                    or item.get("rows")
                    or item.get("n")
                )
                if code is not None:
                    formatted.append(f"{code} ({count})" if count is not None else str(code))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                formatted.append(f"{item[0]} ({item[1]})")
            else:
                formatted.append(str(item))
    else:
        return str(parsed)
    return "; ".join(formatted)


def clean_label(value: Any, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def representative_hashes(value: Any, max_hashes: int = 5) -> str:
    parsed = parse_jsonish(value)
    hashes = []
    if isinstance(parsed, list):
        for item in parsed[:max_hashes]:
            if isinstance(item, dict) and "utterance" in item:
                hashes.append(sha256_text(normalize_segment_text(item["utterance"])))
    return "; ".join(hashes)


def build_cluster_metadata(cluster_summary_path: Path, meta_map_path: Path) -> pd.DataFrame:
    clusters = pd.read_csv(cluster_summary_path)
    require_columns(
        clusters,
        [
            "cluster_id",
            "rows",
            "reviews",
            "papers",
            "mean_score_reviews",
            "accept_rate_papers",
            "top_codes_json",
            "samples_json",
        ],
        str(cluster_summary_path),
    )
    meta_map = load_meta_map(meta_map_path)
    meta_lookup = meta_map.set_index("cluster_id").to_dict(orient="index")
    rows = []
    for internal_row in clusters.itertuples(index=False):
        row_data = internal_row._asdict()
        cluster_id = int(row_data["cluster_id"])
        meta_info = meta_lookup.get(cluster_id, {})
        label = clean_label(
            meta_info.get("display_label"),
            clean_label(
                row_data.get(f"hdbscan_label_{CLUSTERING_CONFIG_ID}"),
                clean_label(
                    row_data.get("cluster_name"),
                    "Other" if cluster_id == -1 else f"cluster_{cluster_id}",
                ),
            ),
        )
        meta_cluster_label = clean_label(
            meta_info.get("meta_label"),
            "Other" if cluster_id == -1 else "",
        )
        rows.append(
            {
                "axial_cluster_id": cluster_id,
                "axial_cluster_label": label,
                "meta_cluster_id": meta_info.get("meta_cluster", -1 if cluster_id == -1 else ""),
                "meta_cluster_label": meta_cluster_label,
                "n_segments": int(row_data["rows"]),
                "n_reviews": int(row_data["reviews"]),
                "n_papers": int(row_data["papers"]),
                "mean_review_score": row_data["mean_score_reviews"],
                "acceptance_rate": row_data["accept_rate_papers"],
                "top_canonical_codes": format_top_codes(row_data["top_codes_json"]),
                "representative_segment_hashes": representative_hashes(row_data["samples_json"]),
                "representative_snippet": "",
            }
        )
    return pd.DataFrame(rows)[CLUSTER_COLUMNS]


def build_codebook(release: pd.DataFrame) -> pd.DataFrame:
    working = release.copy()
    working["_is_accept"] = working["paper_decision"].map(is_accept_value)
    rows = []
    for (code_id, code), group in working.groupby(["canonical_code_id", "canonical_code"], sort=True):
        paper_accept = group[["paper_id", "_is_accept"]].drop_duplicates("paper_id")
        variants = sorted(
            {
                str(value).strip()
                for value in group["raw_open_code"].dropna()
                if str(value).strip()
            }
        )
        segment_counts = group["segment_type"].value_counts()
        rows.append(
            {
                "canonical_code_id": code_id,
                "canonical_code": code,
                "raw_code_variants": "; ".join(variants),
                "count_segments": int(len(group)),
                "count_reviews": int(group["review_id"].nunique()),
                "count_papers": int(group["paper_id"].nunique()),
                "years_observed": "; ".join(
                    str(year) for year in sorted(group["year"].dropna().astype(int).unique())
                ),
                "strength_count": int(segment_counts.get("strength", 0)),
                "weakness_count": int(segment_counts.get("weakness", 0)),
                "full_review_count": int(segment_counts.get("full_review", 0)),
                "mean_review_score": group["review_score"].mean(),
                "acceptance_rate": paper_accept["_is_accept"].mean(),
            }
        )
    return pd.DataFrame(rows)[CODEBOOK_COLUMNS]


def write_readme(path: Path) -> None:
    path.write_text(
        """# ICLR Longitudinal Review Code Dataset, 2020--2025

## Description

This release contains derived LLM-generated open and axial codes for ICLR peer reviews from 2020--2025. The dataset provides segment-level generated qualitative codes, semantically consolidated canonical labels, axial cluster assignments, paper-safe entities, and aggregate metadata for longitudinal analysis of peer-review discourse.

## Source

The source material is OpenReview-hosted ICLR peer reviews. This release does not redistribute raw review text by default. Users who need the raw reviews should retrieve them from OpenReview using the provided source identifiers, subject to OpenReview terms and any applicable venue policies.

## Scale

- Years: 2020--2025
- Reviews: 30,000
- Retained review segments: 39,979
- Segment types: `full_review`, `strength`, `weakness`

## Models used

- Candidate coders: Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.2, Zephyr-7B-beta
- Moderator: Mixtral-8x7B-Instruct-v0.1
- Semantic merge embedding: all-MiniLM-L6-v2
- Axial embedding: all-mpnet-base-v2
- Cluster naming: gpt-4o-mini

## Files

- `iclr_review_codes_2020_2025.csv`: one row per retained review segment with source identifiers, segment hashes, generated codes, canonical code IDs, cluster assignments, paper-safe entities, and paper/review metadata.
- `manual_reference_codes.csv`: one row per segment in the 100-review human-coded reference subset, including human reference codes and similarity metrics to generated codes.
- `cluster_metadata.csv`: one row per axial cluster with cluster labels, meta-cluster labels, aggregate outcome metadata, top canonical codes, and representative segment hashes.
- `codebook.csv`: one row per canonical code with raw variants, counts, years observed, section counts, and outcome summaries.
- `release_stats.json`: summary statistics generated by `scripts/build_release_dataset.py`.

## What is included

- OpenReview-derived identifiers
- Paper/review metadata needed for aggregate analysis
- Segment hashes
- Generated open codes
- Canonical code labels and IDs
- Axial cluster labels and IDs
- Paper-safe normalized entities
- Manual reference codes where available

## What is not included

- Raw review text
- Reviewer identities
- Author identities
- Raw paper titles unless explicitly allowed in a separate private/internal file
- Emails or confidential metadata

## Reconstruction

The public files provide source identifiers and SHA256 hashes of normalized review segments. To reconstruct raw text, users should retrieve reviews from OpenReview according to the source terms, apply the same segmentation policy, normalize whitespace, and compare SHA256 hashes against `segment_hash`.

## Ethical use

This dataset is intended for aggregate research on peer-review discourse, qualitative coding, and evaluation of generated code systems. It must not be used to rank reviewers, authors, papers, institutions, or research areas. It must not be used for automated editorial decisions. The generated labels are analytical summaries and should not be treated as ground-truth reviewer intent.

## Limitations

The dataset covers ICLR only. Section structure is imbalanced across years: 2020--2023 mostly contain full-review segments, while explicit strengths and weaknesses are concentrated in 2024--2025. The labels are LLM-generated and are not ground truth. Seven retained rows with missing generated labels are kept for count consistency and marked `uncoded`. The manual reference subset is small and single-annotator.

## Citation

Anonymous Authors. 2026. *Longitudinal Open and Axial Coding of Peer Reviews*. Anonymous submission.
""",
        encoding="utf-8",
    )


def write_stats(path: Path, release: pd.DataFrame, manual: pd.DataFrame, clusters: pd.DataFrame, codebook: pd.DataFrame) -> None:
    stats = {
        "release_name": "ICLR Longitudinal Review Code Dataset, 2020--2025",
        "raw_text_included": "segment_text" in release.columns,
        "n_segments": int(len(release)),
        "n_reviews": int(release["review_id"].nunique()),
        "n_papers": int(release["paper_id"].nunique()),
        "segment_type_counts": {
            key: int(value) for key, value in release["segment_type"].value_counts().to_dict().items()
        },
        "years": [int(year) for year in sorted(release["year"].unique())],
        "n_canonical_codes": int(release["canonical_code_id"].nunique()),
        "n_canonical_codes_excluding_uncoded": int(
            release.loc[release["canonical_code"] != "uncoded", "canonical_code_id"].nunique()
        ),
        "n_axial_clusters": int(release["axial_cluster_id"].nunique()),
        "n_manual_reference_segments": int(len(manual)),
        "n_cluster_metadata_rows": int(len(clusters)),
        "n_codebook_rows": int(len(codebook)),
    }
    path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")


def write_private_with_text(release: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    release.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public derived release files.")
    parser.add_argument("--coded", type=Path, default=Path("outputs_combined_2020_2025/mixtral_moderated_enriched.csv"))
    parser.add_argument("--semmerge", type=Path, default=Path("outputs_combined_2020_2025/combined_open_codes_2020_2025_semmerge.csv"))
    parser.add_argument("--cluster-summary", type=Path, default=Path("outputs_combined_2020_2025/hdbscan_clusters_labeled_mpnet_mcs50_ms10_umap10nn10_md0_0.csv"))
    parser.add_argument("--meta-map", type=Path, default=Path("outputs_combined_2020_2025/hierarchy_meta_mpnet_mcs50_ms10/meta_cluster_map.csv"))
    parser.add_argument("--paper-safe-entities", type=Path, default=Path("outputs_combined_2020_2025/entity_analysis_papersafe/entity_stats.csv"))
    parser.add_argument("--human-anchor", type=Path, default=Path("outputs_combined_2020_2025/human_anchor_scored/human_anchor_scored_rows.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/release"))
    parser.add_argument("--include_raw_text", action="store_true", help="Allow raw segment text in public CSVs. Off by default.")
    parser.add_argument("--write-private-with-text", action="store_true", help="Write a private Parquet copy with segment text.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    release = build_release_dataframe(
        args.coded,
        args.semmerge,
        args.meta_map,
        args.paper_safe_entities,
        args.human_anchor,
        args.include_raw_text,
    )
    manual = build_manual_reference(args.human_anchor, args.include_raw_text)
    clusters = build_cluster_metadata(args.cluster_summary, args.meta_map)
    codebook = build_codebook(release)

    safe_to_csv(release, args.out_dir / "iclr_review_codes_2020_2025.csv", args.include_raw_text)
    safe_to_csv(manual, args.out_dir / "manual_reference_codes.csv", args.include_raw_text)
    safe_to_csv(clusters, args.out_dir / "cluster_metadata.csv", args.include_raw_text)
    safe_to_csv(codebook, args.out_dir / "codebook.csv", args.include_raw_text)
    write_readme(args.out_dir / "README.md")
    write_stats(args.out_dir / "release_stats.json", release, manual, clusters, codebook)

    if args.write_private_with_text:
        private_release = build_release_dataframe(
            args.coded,
            args.semmerge,
            args.meta_map,
            args.paper_safe_entities,
            args.human_anchor,
            True,
        )
        write_private_with_text(
            private_release,
            Path("data/private/iclr_review_codes_2020_2025_with_text.parquet"),
        )

    print(f"Wrote release files to {args.out_dir}")


if __name__ == "__main__":
    main()
