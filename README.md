# ICLR Open Code Demonstration Sample, 2020-2025

## Description

This repository releases a curated demonstration sample of LLM-generated open
codes and semantically consolidated canonical labels for ICLR peer-review
segments. The sample documents the derived-data format used in the associated
anonymous submission. The complete derived corpus, including axial categories,
is withheld during review and is planned for release upon paper acceptance.

## Source

The source material is OpenReview-hosted ICLR peer reviews from 2020-2025. Raw
review text is not redistributed. Source identifiers and normalized segment
hashes are retained so that users may retrieve source reviews from OpenReview,
subject to its terms and venue policies.

## Demonstration Scope

- Records: 1,000 coded review segments
- Reviews represented: 989
- Papers represented: 936
- Canonical open codes represented: 628
- Segment types: `full_review`, `strength`, `weakness`

This is a quality-curated demonstration sample, not a representative sample for
estimating theme prevalence, year-level trends, or outcome relationships.

## Files

- `data/release/sample_review_codes.csv`: one row per sampled segment with
  identifiers, metadata, raw generated open codes, and canonical open codes.
- `data/release/sample_codebook.csv`: canonical open codes occurring in this
  sample with sample-only frequencies and outcome summaries.
- `data/release/sample_stats.json`: sample counts and release-scope statement.
- `data/release/README.md`: file-level release notes and intended-use guidance.

## Models Used

- Candidate coders: Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.2, Zephyr-7B-beta
- Moderator: Mixtral-8x7B-Instruct-v0.1
- Semantic consolidation embedding: all-MiniLM-L6-v2

## Withheld During Review

- Raw review text
- The complete coded corpus and complete codebook
- Axial cluster assignments, axial labels, and meta-category labels
- The manual reference-code evaluation subset
- Reviewer or author identity information

## Ethical Use

The released codes are analytical summaries, not ground-truth reviewer intent.
This sample is intended for inspection of the coding representation and should
not be used to rank reviewers, authors, papers, institutions, or research
areas, or for automated editorial decisions.

## Citation

Anonymous Authors. 2026. *Longitudinal Open and Axial Coding of Peer Reviews*.
Anonymous submission.

## Validation

Run:

```bash
python scripts/validate_release_dataset.py --release-dir data/release
```

The validator checks the sample-only schema, expected counts, canonical-label
formatting, and absence of withheld public-release columns and files.
