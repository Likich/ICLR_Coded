# ICLR Review Code Sample Release, 2020-2025

## Description

This curated demonstration release contains 1,000 retained review-segment
records with generated open codes and semantically consolidated canonical labels
for ICLR reviews. It is provided to document the open-code data format and support
preliminary inspection while the complete derived corpus is withheld during paper
review.

The sample excludes all segments in the manual reference-code evaluation
subset. Records were quality-screened using internal pipeline diagnostics and
code-to-source-utterance coherence checks, then sampled across year and segment
type with diversity caps. Axial codes, internal quality diagnostics, and the
complete corpus are not included in this demonstration release; we plan to
release the full derived data with axial codes upon paper acceptance.
Counts and outcome summaries in the accompanying files are computed from this
curated sample only and are not full-corpus statistics. Because it is selected
for label quality, it is not a representative distributional sample of the
complete corpus.

## Files

- `sample_review_codes.csv`: 1,000 sampled segment records with generated raw and
  canonical open codes and review-level metadata.
- `sample_codebook.csv`: the 628 canonical codes occurring in the sample,
  with sample-only frequencies and outcome summaries.
- `sample_stats.json`: sample-level counts and release-scope information.

## Exclusions

- Raw review text is not redistributed.
- The 100-review manual reference subset is not included.
- Axial code assignments, axial labels, and cluster metadata are not included.
- The full codebook and full segment-level release are not included.

## Intended use

This sample supports inspection of the data schema and derived labels. It must
not be used to rank reviewers, papers, authors, institutions, or research
areas, or as a substitute for human peer review.
