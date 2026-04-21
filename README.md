# BibTeX Fixer

This repository provides a small command-line script that checks BibTeX
entries against online bibliographic sources and rewrites a cleaned,
normalized BibTeX file.

## What the script does

- Parses BibTeX entries from an input file.
- Uses DOI-first lookup when a DOI is present, whether it appears in an
  explicit `doi` field or a DOI URL.
- Looks up entries by title using Crossref, OpenAlex, Semantic Scholar, and
  OpenReview when venue metadata is incomplete or an arXiv-tagged entry needs
  to be promoted to its published version.
- Treats arXiv metadata as a fallback only: arXiv-tagged entries are upgraded
  to a published venue record when one is found, and arXiv is queried only if
  no published version can be resolved.
- Merges fetched metadata into the existing fields.
- Writes a normalized BibTeX file with a consistent field order.
- Writes separate BibTeX and log files for entries that remain unresolved.
- Supports retrying only previously unresolved entries from a past run.
- Retries transient HTTP failures with configurable timeout, retry count, and
  backoff settings.

## Usage

Basic usage:

```bash
python check_and_fix_biblio.py \
  --input biblio.bib \
  --output biblio_fixed.bib
```

Main options:

- `--mailto`: Email address to include in the Crossref User-Agent header.
- `--min-similarity`: Minimum title similarity (0-1) required to accept a match.
- `--crossref-rows`: Number of Crossref results to consider per entry.
- `--openalex-rows`: Number of OpenAlex results to consider per entry.
- `--semantic-scholar-rows`: Number of Semantic Scholar results to consider per entry.
- `--delay`: Delay in seconds between requests to external services.
- `--limit`: Process only the first N entries (leave the rest unchanged).
- `--retry-unresolved`: Recheck only the keys listed in a previous unresolved
  BibTeX file and merge the results into the full output file.
- `--unresolved-output`: Path for the BibTeX subset of entries that still
  could not be resolved.
- `--unresolved-log`: Path for a human-readable log of unresolved entries and
  failure reasons.
- `--http-timeout`: Per-request timeout for metadata lookups.
- `--http-retries`: Number of retries for transient HTTP failures such as
  `429` and `503`.
- `--retry-backoff`: Base exponential backoff in seconds between retries.

Example with custom settings:

```bash
python check_and_fix_biblio.py \
  --input my_refs.bib \
  --output my_refs_checked.bib \
  --mailto you@example.com \
  --min-similarity 0.9 \
  --delay 1.0
```

Retry only entries that were unresolved in a previous run:

```bash
python check_and_fix_biblio.py \
  --input my_refs.bib \
  --output my_refs_checked_retry.bib \
  --retry-unresolved unresolved-bib-entries.bib \
  --unresolved-output unresolved-bib-entries-next.bib \
  --unresolved-log unresolved-bib-entries-next.log \
  --http-timeout 60
```

## Outputs

- `--output` contains the full rewritten bibliography.
- `--unresolved-output` contains only the entries that remain unresolved after
  the run. This file can be passed back to `--retry-unresolved`.
- `--unresolved-log` contains a timestamped summary of unresolved entries,
  their reasons, and the original BibTeX text.

## Notes

- Matching is **title-based**; author/year are not used for disambiguation.
- Entries tagged as arXiv through fields such as `journal`, `eprint`,
  `archiveprefix`, `url`, or `note` are treated as preprints and are upgraded
  to published metadata when a non-arXiv match is found.
- When a published venue is found for an arXiv preprint, arXiv-specific fields
  such as `eprint`, `archiveprefix`, `primaryclass`, arXiv URLs, and arXiv
  notes are removed.
- `--retry-unresolved` only rechecks keys present in the retry file; all other
  entries from `--input` are copied through unchanged.
- Entries are reformatted; original comments and non-entry content are not
  preserved.
- Output is ASCII-only; non-ASCII characters are stripped.
