# BibTeX Fixer

This repository provides a small command-line script that checks BibTeX
entries against online bibliographic sources and rewrites a cleaned,
normalized BibTeX file.

## What the script does

- Parses BibTeX entries from an input file.
- Uses DOI-first lookup when a DOI is present, then falls back to title search.
- Looks up entries by title using Crossref, OpenAlex, Semantic Scholar, and OpenReview.
- For arXiv preprints, it prefers a published venue when one is found (including OpenReview for ICLR-style venues).
- Merges fetched metadata into the existing fields.
- Writes a normalized BibTeX file with a consistent field order.

## Requirements

- Python 3.8+ (stdlib only; no external dependencies)
- Network access (Crossref, OpenAlex, Semantic Scholar, OpenReview, arXiv)

## Usage

Basic usage:

```bash
python check_and_fix_biblio.py  --input biblio.bib  --output biblio_fixed.bib
```

Common options:

- `--mailto`: Email address to include in the Crossref User-Agent header.
- `--min-similarity`: Minimum title similarity (0-1) required to accept a match.
- `--crossref-rows`: Number of Crossref results to consider per entry.
- `--openalex-rows`: Number of OpenAlex results to consider per entry.
- `--semantic-scholar-rows`: Number of Semantic Scholar results to consider per entry.
- `--delay`: Delay in seconds between requests to external services.
- `--limit`: Process only the first N entries (leave the rest unchanged).

Example with custom settings:

```bash
python check_and_fix_biblio.py \
  --input my_refs.bib \
  --output my_refs_checked.bib \
  --mailto you@example.com \
  --min-similarity 0.9 \
  --delay 1.0
```

## Output

The script prints a per-entry status and a summary of how many entries were
updated, unchanged, unresolved, or had errors. The output file is rewritten
with a consistent field order and ASCII-only content.

## Notes and limitations

- Matching is title-based; author/year are not used for disambiguation.
- Entries are reformatted; original comments and non-entry content are not
  preserved.
- Output is ASCII-only; non-ASCII characters are stripped.
- When a published venue is found for an arXiv preprint, arXiv-specific fields
  (eprint/archiveprefix/primaryclass and arXiv URLs) are removed.
