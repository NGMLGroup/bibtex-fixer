#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher


def http_get(url, headers=None, timeout=30):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def clean_whitespace(value):
    return re.sub(r"\s+", " ", value or "").strip()


def ascii_safe(value):
    if value is None:
        return None
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    value = value.replace("\u00a0", " ")
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    return clean_whitespace(value)


def normalize_title(value):
    value = value or ""
    value = value.lower()
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = re.sub(r"[{}\"']", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def normalize_value(value):
    value = value or ""
    value = value.lower()
    value = re.sub(r"[{}\"']", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def title_similarity(a, b):
    if not a or not b:
        return 0.0
    a_norm = normalize_title(a)
    b_norm = normalize_title(b)
    if not a_norm or not b_norm:
        return 0.0
    seq_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if a_tokens and b_tokens:
        jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
    else:
        jaccard = 0.0
    return max(seq_ratio, jaccard)


def split_entries(text):
    entries = []
    i = 0
    while i < len(text):
        if text[i] != "@":
            i += 1
            continue
        start = i
        j = i + 1
        while j < len(text) and text[j] not in "{(":
            j += 1
        if j >= len(text):
            break
        open_char = text[j]
        close_char = "}" if open_char == "{" else ")"
        level = 0
        k = j
        while k < len(text):
            char = text[k]
            if char == open_char:
                level += 1
            elif char == close_char:
                level -= 1
                if level == 0:
                    k += 1
                    entries.append(text[start:k].strip())
                    i = k
                    break
            k += 1
        else:
            break
    return entries


def split_top_level(text):
    parts = []
    current = []
    level = 0
    in_quote = False
    escape = False
    for char in text:
        if in_quote:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_quote = False
            continue
        if char == '"':
            in_quote = True
            current.append(char)
            continue
        if char == "{":
            level += 1
        elif char == "}":
            if level > 0:
                level -= 1
        if char == "," and level == 0 and not in_quote:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def is_wrapped(value, open_char, close_char):
    if not (value.startswith(open_char) and value.endswith(close_char)):
        return False
    level = 0
    for idx, char in enumerate(value):
        if char == open_char:
            level += 1
        elif char == close_char:
            level -= 1
            if level == 0 and idx != len(value) - 1:
                return False
    return level == 0


def strip_wrapping(value):
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].strip()
    if value.startswith("{") and value.endswith("}") and is_wrapped(value, "{", "}"):
        return value[1:-1].strip()
    return value


def parse_entry(entry_text):
    match = re.match(r"@\s*([^{(\s]+)\s*([({])", entry_text)
    if not match:
        return None, None, {}
    entry_type = match.group(1).strip().lower()
    open_char = match.group(2)
    close_char = "}" if open_char == "{" else ")"
    inside = entry_text[match.end():].strip()
    if inside.endswith(close_char):
        inside = inside[:-1].strip()
    parts = split_top_level(inside)
    if not parts:
        return entry_type, None, {}
    key = parts[0].strip().rstrip(",")
    fields = {}
    for part in parts[1:]:
        if not part or "=" not in part:
            continue
        field, value = part.split("=", 1)
        field = field.strip().lower()
        value = strip_wrapping(value.strip())
        fields[field] = value
    return entry_type, key, fields


def format_entry(entry_type, key, fields):
    ordered_fields = [
        "title",
        "author",
        "editor",
        "journal",
        "booktitle",
        "year",
        "volume",
        "number",
        "pages",
        "publisher",
        "series",
        "doi",
        "url",
        "eprint",
        "archiveprefix",
        "primaryclass",
        "note",
    ]
    items = []
    for field in ordered_fields:
        value = fields.get(field)
        if value:
            items.append((field, value))
    for field in sorted(fields.keys()):
        if field in ordered_fields:
            continue
        value = fields.get(field)
        if value:
            items.append((field, value))
    lines = [f"@{entry_type}{{{key},"]
    for idx, (field, value) in enumerate(items):
        comma = "," if idx < len(items) - 1 else ""
        lines.append(f"  {field} = {{{value}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def is_arxiv_entry(fields):
    for field in ("journal", "eprint", "archiveprefix", "url", "note", "doi"):
        value = fields.get(field, "")
        if "arxiv" in value.lower():
            return True
    return False


def extract_doi(fields):
    for field in ("doi", "url"):
        value = fields.get(field)
        if not value:
            continue
        value = value.strip()
        if not value:
            continue
        lower_value = value.lower()
        if "doi.org/" in lower_value:
            doi = value.split("doi.org/", 1)[1].strip()
            if doi:
                return doi.rstrip(".")
        match = re.search(r"(10\.\d{4,9}/[^\s{}<>]+)", value, re.IGNORECASE)
        if match:
            return match.group(1).rstrip(".")
    return None


def remove_arxiv_fields(fields):
    cleaned = dict(fields)
    for field in ("eprint", "archiveprefix", "primaryclass"):
        cleaned.pop(field, None)
    for field in ("journal", "url", "note"):
        value = cleaned.get(field)
        if value and "arxiv" in value.lower():
            cleaned.pop(field, None)
    return cleaned


def crossref_item_is_preprint(item):
    item_type = (item.get("type") or "").lower()
    if item_type in ("posted-content", "preprint"):
        return True
    container = (item.get("container-title") or [None])[0] or ""
    if "arxiv" in container.lower():
        return True
    url = item.get("URL") or ""
    if "arxiv" in url.lower():
        return True
    return False


def openalex_item_is_preprint(item):
    item_type = (item.get("type") or "").lower()
    if item_type == "preprint":
        return True
    host = item.get("host_venue") or {}
    venue = host.get("display_name") or ""
    return "arxiv" in venue.lower()


def semantic_scholar_item_is_preprint(item):
    journal_name = (item.get("journal") or {}).get("name") or ""
    venue = item.get("venue") or ""
    publication_venue = (item.get("publicationVenue") or {}).get("name") or ""
    if (
        "arxiv" in journal_name.lower()
        or "arxiv" in venue.lower()
        or "arxiv" in publication_venue.lower()
    ):
        return True
    types = item.get("publicationTypes") or []
    if any(entry_type.lower() == "preprint" for entry_type in types):
        return True
    external_ids = item.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv") or external_ids.get("arXiv")
    if arxiv_id and not (journal_name or venue or publication_venue):
        return True
    return False


def first_author_for_paperhash(author_field):
    if not author_field:
        return None
    first_author = author_field.split(" and ", 1)[0].strip()
    if not first_author:
        return None
    if "," in first_author:
        parts = [part.strip() for part in first_author.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            first_author = f"{parts[1]} {parts[0]}"
        else:
            first_author = first_author.replace(",", " ")
    return clean_whitespace(first_author)


def author_looks_incomplete(author_field):
    if not author_field:
        return True
    author_lower = author_field.lower()
    if "et al" in author_lower:
        return True
    return False


def get_paperhash(first_author, title):
    title = title.strip()
    strip_punctuation = r"[^A-Za-z\d\s]"
    title = re.sub(strip_punctuation, "", title)
    first_author = re.sub(strip_punctuation, "", first_author)
    first_author = first_author.split(" ").pop()
    title = re.sub(strip_punctuation, "", title)
    title = re.sub(r"\r|\n", "", title)
    title = re.sub(r"\s+", "_", title)
    first_author = re.sub(strip_punctuation, "", first_author)
    return (first_author + "|" + title).lower()


def openreview_value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def openreview_item_title(item):
    content = item.get("content") or {}
    forum_content = item.get("forumContent") or {}
    return openreview_value(content.get("title")) or openreview_value(
        forum_content.get("title")
    )


def openreview_item_priority(item):
    invitations = item.get("invitations") or item.get("invitation") or []
    if isinstance(invitations, str):
        invitations = [invitations]
    content = item.get("content") or {}
    venueid = openreview_value(content.get("venueid")) or ""
    for invitation in invitations:
        invitation_lower = invitation.lower()
        if (
            "/-/submission" in invitation_lower
            or "/-/camera_ready" in invitation_lower
            or "/-/paper" in invitation_lower
        ):
            return 3
        if ".cc/" in invitation or "openreview.net" in invitation:
            return 2
    if ".cc/" in venueid:
        return 1
    return 0


def openreview_item_is_preprint(item):
    content = item.get("content") or {}
    forum_content = item.get("forumContent") or {}
    venue = openreview_value(content.get("venue")) or openreview_value(
        forum_content.get("venue")
    )
    venueid = openreview_value(content.get("venueid")) or openreview_value(
        forum_content.get("venueid")
    )
    for value in (venue, venueid):
        if value and any(token in value.lower() for token in ("arxiv", "corr")):
            return True
    return False


def extract_year_from_text(value):
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", value)
    if match:
        return int(match.group(0))
    return None


def ordinal_word(number):
    ones = {
        1: "First",
        2: "Second",
        3: "Third",
        4: "Fourth",
        5: "Fifth",
        6: "Sixth",
        7: "Seventh",
        8: "Eighth",
        9: "Ninth",
        10: "Tenth",
        11: "Eleventh",
        12: "Twelfth",
        13: "Thirteenth",
        14: "Fourteenth",
        15: "Fifteenth",
        16: "Sixteenth",
        17: "Seventeenth",
        18: "Eighteenth",
        19: "Nineteenth",
    }
    tens_ordinal = {
        20: "Twentieth",
        30: "Thirtieth",
        40: "Fortieth",
        50: "Fiftieth",
        60: "Sixtieth",
        70: "Seventieth",
        80: "Eightieth",
        90: "Ninetieth",
    }
    tens_cardinal = {
        2: "Twenty",
        3: "Thirty",
        4: "Forty",
        5: "Fifty",
        6: "Sixty",
        7: "Seventy",
        8: "Eighty",
        9: "Ninety",
    }
    if number in ones:
        return ones[number]
    if number in tens_ordinal:
        return tens_ordinal[number]
    if 20 < number < 100:
        tens, ones_digit = divmod(number, 10)
        tens_word = tens_cardinal.get(tens)
        ones_word = ones.get(ones_digit)
        if tens_word and ones_word:
            return f"{tens_word}-{ones_word.lower()}"
    return None


def iclr_booktitle(year):
    if not year:
        return "International Conference on Learning Representations"
    ordinal = year - 2012
    word = ordinal_word(ordinal)
    if word:
        return f"The {word} International Conference on Learning Representations"
    return f"The {ordinal}th International Conference on Learning Representations"


def icml_booktitle(year):
    if not year:
        return "International Conference on Machine Learning"
    ordinal = year - 1983
    word = ordinal_word(ordinal)
    if word:
        return f"{word} International Conference on Machine Learning"
    return f"{ordinal}th International Conference on Machine Learning"


def neurips_booktitle(year):
    if not year:
        return "Conference on Neural Information Processing Systems"
    ordinal = year - 1986
    word = ordinal_word(ordinal)
    if word:
        return f"{word} Conference on Neural Information Processing Systems"
    return f"{ordinal}th Conference on Neural Information Processing Systems"


def openreview_venue_to_booktitle(venue):
    if not venue:
        return None
    venue = clean_whitespace(str(venue))
    venue_lower = venue.lower()
    for suffix in (" poster", " spotlight", " oral"):
        if venue_lower.endswith(suffix):
            venue = venue[: -len(suffix)]
            venue_lower = venue.lower()
            break
    venue_lower = venue.lower()
    if "iclr" in venue_lower or "iclr.cc" in venue_lower:
        year = extract_year_from_text(venue)
        return iclr_booktitle(year)
    if "icml" in venue_lower or "icml.cc" in venue_lower:
        year = extract_year_from_text(venue)
        return icml_booktitle(year)
    if "neurips" in venue_lower or "nips" in venue_lower:
        year = extract_year_from_text(venue)
        return neurips_booktitle(year)
    return venue


def openreview_date_to_year(value):
    if value is None:
        return None
    try:
        timestamp = int(value) / 1000
    except (ValueError, TypeError):
        return None
    return time.gmtime(timestamp).tm_year


def search_openreview_paperhash(title, author_field, rows=5):
    first_author = first_author_for_paperhash(author_field)
    if not title or not first_author:
        return []
    paperhash = get_paperhash(first_author, title)
    params = {"paperhash": paperhash, "limit": rows}
    url = "https://api2.openreview.net/notes?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "check_and_fix_biblio/0.1"}
    data = json.loads(http_get(url, headers=headers).decode("utf-8"))
    return data.get("notes", [])


def search_openreview_by_title(title, rows=10):
    if not title:
        return []
    params = {"term": title, "limit": rows}
    url = "https://api2.openreview.net/notes/search?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "check_and_fix_biblio/0.1"}
    data = json.loads(http_get(url, headers=headers).decode("utf-8"))
    return data.get("notes", [])


def search_openreview(title, author_field, rows=5):
    candidates = []
    if not author_looks_incomplete(author_field):
        candidates = search_openreview_paperhash(title, author_field, rows=rows)
    if candidates:
        return candidates
    return search_openreview_by_title(title, rows=max(10, rows))


def find_best_openreview_match(title, candidates, min_similarity, skip_predicate=None):
    best_item = None
    best_score = 0.0
    best_priority = -1
    for item in candidates:
        if skip_predicate and skip_predicate(item):
            continue
        candidate_title = openreview_item_title(item)
        if not candidate_title:
            continue
        score = title_similarity(title, candidate_title)
        if score < min_similarity:
            continue
        priority = openreview_item_priority(item)
        if score > best_score or (score == best_score and priority > best_priority):
            best_item = item
            best_score = score
            best_priority = priority
    return best_item, best_score


def openreview_item_to_fields(item):
    fields = {}
    content = item.get("content") or {}
    forum_content = item.get("forumContent") or {}

    def get_field(name):
        return openreview_value(content.get(name)) or openreview_value(
            forum_content.get(name)
        )

    title = get_field("title")
    if title:
        fields["title"] = title
    authors = get_field("authors")
    if authors:
        fields["author"] = " and ".join(authors)
    year = get_field("year")
    if not year:
        year = openreview_date_to_year(item.get("pdate") or item.get("cdate"))
    if year:
        fields["year"] = str(year)
    venue = get_field("venue")
    venueid = get_field("venueid")
    booktitle = openreview_venue_to_booktitle(venue) or openreview_venue_to_booktitle(
        venueid
    )
    if booktitle:
        fields["booktitle"] = booktitle
    forum_id = item.get("forum") or item.get("id")
    if forum_id:
        fields["url"] = f"https://openreview.net/forum?id={forum_id}"
    return fields


def search_crossref(title, rows=5, mailto=None):
    params = {"query.title": title, "rows": rows}
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    user_agent = "check_and_fix_biblio/0.1"
    if mailto:
        user_agent = f"{user_agent} (mailto:{mailto})"
    headers = {"User-Agent": user_agent}
    data = json.loads(http_get(url, headers=headers).decode("utf-8"))
    return data.get("message", {}).get("items", [])


def search_openalex(title, rows=10, mailto=None):
    params = {"search": title, "per-page": rows}
    if mailto:
        params["mailto"] = mailto
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    user_agent = "check_and_fix_biblio/0.1"
    headers = {"User-Agent": user_agent}
    data = json.loads(http_get(url, headers=headers).decode("utf-8"))
    return data.get("results", [])


def openalex_item_to_fields(item, entry_type):
    fields = {}
    title = item.get("title") or item.get("display_name")
    if title:
        fields["title"] = title
    authors = []
    for authorship in item.get("authorships", []) or []:
        author = authorship.get("author") or {}
        name = author.get("display_name") or author.get("name")
        if name:
            authors.append(name)
    if authors:
        fields["author"] = " and ".join(authors)
    year = item.get("publication_year")
    if not year and item.get("publication_date"):
        year = item["publication_date"].split("-")[0]
    if year:
        fields["year"] = str(year)
    host = item.get("host_venue") or {}
    venue = host.get("display_name")
    if venue:
        if entry_type == "inproceedings":
            fields["booktitle"] = venue
        else:
            fields["journal"] = venue
    biblio = item.get("biblio") or {}
    volume = biblio.get("volume")
    issue = biblio.get("issue")
    first_page = biblio.get("first_page")
    last_page = biblio.get("last_page")
    if volume:
        fields["volume"] = str(volume)
    if issue:
        fields["number"] = str(issue)
    if first_page and last_page:
        fields["pages"] = f"{first_page}-{last_page}"
    elif first_page:
        fields["pages"] = str(first_page)
    doi = item.get("doi")
    if doi:
        doi = doi.replace("https://doi.org/", "")
        fields["doi"] = doi
        fields["url"] = f"https://doi.org/{doi}"
    return fields


def openalex_type_to_bibtex(entry_type, fallback):
    mapping = {
        "journal-article": "article",
        "proceedings-article": "inproceedings",
        "book-chapter": "incollection",
        "book": "book",
        "report": "techreport",
        "preprint": "article",
    }
    return mapping.get(entry_type, fallback or "article")


def search_semantic_scholar(title, rows=10):
    params = {
        "query": title,
        "limit": rows,
        "fields": (
            "title,authors,year,venue,journal,externalIds,url,"
            "publicationTypes,publicationVenue"
        ),
    }
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        params
    )
    headers = {"User-Agent": "check_and_fix_biblio/0.1"}
    data = json.loads(http_get(url, headers=headers).decode("utf-8"))
    return data.get("data", [])


def semantic_scholar_item_to_fields(item, entry_type):
    fields = {}
    title = item.get("title")
    if title:
        fields["title"] = title
    authors = []
    for author in item.get("authors", []) or []:
        name = author.get("name")
        if name:
            authors.append(name)
    if authors:
        fields["author"] = " and ".join(authors)
    year = item.get("year")
    if year:
        fields["year"] = str(year)
    journal = item.get("journal") or {}
    journal_name = journal.get("name")
    venue = item.get("venue")
    publication_venue = item.get("publicationVenue") or {}
    if not venue:
        venue = publication_venue.get("name")
    if journal_name:
        fields["journal"] = journal_name
        if journal.get("volume"):
            fields["volume"] = str(journal["volume"])
        if journal.get("pages"):
            fields["pages"] = str(journal["pages"])
    elif venue:
        if entry_type == "inproceedings":
            fields["booktitle"] = venue
        else:
            fields["journal"] = venue
    external_ids = item.get("externalIds") or {}
    doi = external_ids.get("DOI") or external_ids.get("doi")
    if doi:
        fields["doi"] = doi
        fields["url"] = f"https://doi.org/{doi}"
    elif item.get("url"):
        fields["url"] = item["url"]
    arxiv_id = external_ids.get("ArXiv") or external_ids.get("arXiv")
    if arxiv_id and not (journal_name or venue):
        fields["journal"] = f"arXiv preprint arXiv:{arxiv_id}"
        fields["eprint"] = arxiv_id
        fields["archiveprefix"] = "arXiv"
    return fields


def semantic_scholar_type_to_bibtex(item, fallback):
    types = item.get("publicationTypes") or []
    mapping = {
        "Conference": "inproceedings",
        "JournalArticle": "article",
        "Review": "article",
        "Book": "book",
        "BookChapter": "incollection",
        "Preprint": "article",
    }
    for entry_type in types:
        mapped = mapping.get(entry_type)
        if mapped:
            return mapped
    return fallback or "article"


def fetch_crossref_bibtex(doi, mailto=None):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}/transform/application/x-bibtex"
    user_agent = "check_and_fix_biblio/0.1"
    if mailto:
        user_agent = f"{user_agent} (mailto:{mailto})"
    headers = {"User-Agent": user_agent, "Accept": "application/x-bibtex"}
    return http_get(url, headers=headers).decode("utf-8")


def crossref_item_to_fields(item, entry_type):
    fields = {}
    title = (item.get("title") or [None])[0]
    if title:
        fields["title"] = title
    authors = []
    for author in item.get("author", []) or []:
        family = author.get("family")
        given = author.get("given")
        if family and given:
            authors.append(f"{family}, {given}")
        elif author.get("name"):
            authors.append(author["name"])
        elif family:
            authors.append(family)
    if authors:
        fields["author"] = " and ".join(authors)
    issued = item.get("issued", {}).get("date-parts", [])
    if issued and issued[0]:
        fields["year"] = str(issued[0][0])
    container = (item.get("container-title") or [None])[0]
    if container:
        if entry_type == "inproceedings":
            fields["booktitle"] = container
        else:
            fields["journal"] = container
    if item.get("volume"):
        fields["volume"] = str(item["volume"])
    if item.get("issue"):
        fields["number"] = str(item["issue"])
    if item.get("page"):
        fields["pages"] = str(item["page"])
    if item.get("publisher"):
        fields["publisher"] = item["publisher"]
    if item.get("DOI"):
        fields["doi"] = item["DOI"]
    if item.get("URL"):
        fields["url"] = item["URL"]
    return fields


def crossref_type_to_bibtex(entry_type, fallback):
    mapping = {
        "journal-article": "article",
        "proceedings-article": "inproceedings",
        "book-chapter": "incollection",
        "book": "book",
        "proceedings": "proceedings",
        "report": "techreport",
    }
    return mapping.get(entry_type, fallback or "article")


def search_arxiv(title, max_results=5):
    query = f'ti:"{title}"'
    params = {"search_query": query, "start": 0, "max_results": max_results}
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    xml_data = http_get(url).decode("utf-8")
    root = ET.fromstring(xml_data)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    results = []
    for entry in root.findall("atom:entry", ns):
        title_elem = entry.find("atom:title", ns)
        if title_elem is None or not title_elem.text:
            continue
        entry_title = clean_whitespace(title_elem.text)
        published_elem = entry.find("atom:published", ns)
        published = published_elem.text if published_elem is not None else ""
        authors = []
        for author in entry.findall("atom:author", ns):
            name_elem = author.find("atom:name", ns)
            if name_elem is not None and name_elem.text:
                authors.append(clean_whitespace(name_elem.text))
        id_elem = entry.find("atom:id", ns)
        arxiv_id = None
        if id_elem is not None and id_elem.text:
            arxiv_id = id_elem.text.split("/abs/")[-1]
        primary = entry.find("arxiv:primary_category", ns)
        primary_class = primary.attrib.get("term") if primary is not None else None
        results.append(
            {
                "title": entry_title,
                "published": published,
                "authors": authors,
                "arxiv_id": arxiv_id,
                "primary_class": primary_class,
            }
        )
    return results


def arxiv_entry_to_fields(item):
    fields = {}
    fields["title"] = item["title"]
    if item.get("authors"):
        fields["author"] = " and ".join(item["authors"])
    year = None
    if item.get("published"):
        year = item["published"].split("-")[0]
    if year:
        fields["year"] = year
    arxiv_id = item.get("arxiv_id")
    if arxiv_id:
        fields["journal"] = f"arXiv preprint arXiv:{arxiv_id}"
        fields["url"] = f"https://arxiv.org/abs/{arxiv_id}"
        fields["eprint"] = arxiv_id
        fields["archiveprefix"] = "arXiv"
    if item.get("primary_class"):
        fields["primaryclass"] = item["primary_class"]
    return fields


def sanitize_fields(fields):
    sanitized = {}
    for field, value in fields.items():
        if value is None:
            continue
        value = ascii_safe(str(value))
        if value:
            sanitized[field] = value
    return sanitized


def merge_fields(existing, candidate):
    merged = dict(existing)
    for field, value in candidate.items():
        if value:
            merged[field] = value
    return merged


def detect_changes(existing, updated):
    changed = []
    for field, value in updated.items():
        if normalize_value(existing.get(field, "")) != normalize_value(value):
            changed.append(field)
    for field in existing.keys():
        if field not in updated:
            changed.append(field)
    return sorted(set(changed))


def find_best_match_by_title(title, candidates, min_similarity, skip_predicate=None):
    best_item = None
    best_score = 0.0
    for item in candidates:
        if skip_predicate and skip_predicate(item):
            continue
        candidate_title = item.get("title")
        if isinstance(candidate_title, list):
            candidate_title = candidate_title[0] if candidate_title else None
        if not candidate_title:
            candidate_title = item.get("display_name")
        if not candidate_title:
            continue
        score = title_similarity(title, candidate_title)
        if score > best_score:
            best_score = score
            best_item = item
    if best_item and best_score >= min_similarity:
        return best_item, best_score
    return None, best_score


def process_entry(
    entry_type,
    key,
    fields,
    min_similarity,
    mailto,
    delay,
    crossref_rows,
    openalex_rows,
    semantic_scholar_rows,
):
    errors = []
    scores = {}
    entry_is_arxiv = is_arxiv_entry(fields)
    doi_fallback = None
    doi = extract_doi(fields)
    if doi:
        try:
            bibtex = fetch_crossref_bibtex(doi, mailto=mailto)
            parsed_type, _, parsed_fields = parse_entry(bibtex)
            candidate_type = parsed_type or entry_type or "article"
            candidate_fields = sanitize_fields(parsed_fields)
            candidate_is_arxiv = is_arxiv_entry(candidate_fields)
            if not entry_is_arxiv or not candidate_is_arxiv:
                time.sleep(delay)
                return (candidate_type, candidate_fields, candidate_is_arxiv), None
            doi_fallback = (candidate_type, candidate_fields, candidate_is_arxiv)
        except Exception as exc:
            errors.append(f"Crossref DOI lookup failed: {exc}")
        time.sleep(delay)

    title = fields.get("title")
    if not title:
        if doi_fallback:
            return doi_fallback, None
        if errors:
            return None, f"missing title; {'; '.join(errors)}"
        return None, "missing title"

    should_try_openreview = (
        entry_is_arxiv
        or not (fields.get("booktitle") or fields.get("journal"))
        or not (fields.get("url") or fields.get("doi"))
        or author_looks_incomplete(fields.get("author"))
    )
    if should_try_openreview:
        try:
            items = search_openreview(title, fields.get("author"), rows=10)
            best_item, score = find_best_openreview_match(
                title,
                items,
                min_similarity,
                skip_predicate=openreview_item_is_preprint if entry_is_arxiv else None,
            )
            scores["OpenReview"] = score
            if best_item:
                candidate_fields = openreview_item_to_fields(best_item)
                candidate_fields = sanitize_fields(candidate_fields)
                candidate_type = entry_type or "article"
                if "booktitle" in candidate_fields:
                    candidate_type = "inproceedings"
                candidate_is_arxiv = is_arxiv_entry(candidate_fields)
                time.sleep(delay)
                return (candidate_type, candidate_fields, candidate_is_arxiv), None
        except Exception as exc:
            errors.append(f"OpenReview lookup failed: {exc}")
        time.sleep(delay)

    try:
        items = search_crossref(title, rows=crossref_rows, mailto=mailto)
        best_item, score = find_best_match_by_title(
            title,
            items,
            min_similarity,
            skip_predicate=crossref_item_is_preprint if entry_is_arxiv else None,
        )
        scores["Crossref"] = score
        if best_item:
            candidate_type = crossref_type_to_bibtex(best_item.get("type"), entry_type)
            candidate_fields = None
            doi = best_item.get("DOI")
            if doi:
                try:
                    bibtex = fetch_crossref_bibtex(doi, mailto=mailto)
                    parsed_type, _, parsed_fields = parse_entry(bibtex)
                    candidate_type = parsed_type or candidate_type
                    candidate_fields = parsed_fields
                except Exception:
                    candidate_fields = crossref_item_to_fields(best_item, candidate_type)
            else:
                candidate_fields = crossref_item_to_fields(best_item, candidate_type)
            candidate_fields = sanitize_fields(candidate_fields)
            candidate_is_arxiv = is_arxiv_entry(candidate_fields)
            if not (entry_is_arxiv and candidate_is_arxiv):
                time.sleep(delay)
                return (candidate_type, candidate_fields, candidate_is_arxiv), None
    except Exception as exc:
        errors.append(f"Crossref lookup failed: {exc}")
    time.sleep(delay)

    try:
        items = search_openalex(title, rows=openalex_rows, mailto=mailto)
        best_item, score = find_best_match_by_title(
            title,
            items,
            min_similarity,
            skip_predicate=openalex_item_is_preprint if entry_is_arxiv else None,
        )
        scores["OpenAlex"] = score
        if best_item:
            candidate_type = openalex_type_to_bibtex(best_item.get("type"), entry_type)
            candidate_fields = openalex_item_to_fields(best_item, candidate_type)
            candidate_fields = sanitize_fields(candidate_fields)
            candidate_is_arxiv = is_arxiv_entry(candidate_fields)
            if not (entry_is_arxiv and candidate_is_arxiv):
                time.sleep(delay)
                return (candidate_type, candidate_fields, candidate_is_arxiv), None
    except Exception as exc:
        errors.append(f"OpenAlex lookup failed: {exc}")
    time.sleep(delay)

    try:
        items = search_semantic_scholar(title, rows=semantic_scholar_rows)
        best_item, score = find_best_match_by_title(
            title,
            items,
            min_similarity,
            skip_predicate=semantic_scholar_item_is_preprint if entry_is_arxiv else None,
        )
        scores["Semantic Scholar"] = score
        if best_item:
            candidate_type = semantic_scholar_type_to_bibtex(best_item, entry_type)
            candidate_fields = semantic_scholar_item_to_fields(best_item, candidate_type)
            candidate_fields = sanitize_fields(candidate_fields)
            candidate_is_arxiv = is_arxiv_entry(candidate_fields)
            if not (entry_is_arxiv and candidate_is_arxiv):
                time.sleep(delay)
                return (candidate_type, candidate_fields, candidate_is_arxiv), None
    except Exception as exc:
        errors.append(f"Semantic Scholar lookup failed: {exc}")
    time.sleep(delay)

    if entry_is_arxiv:
        if doi_fallback:
            return doi_fallback, None
        try:
            candidates = search_arxiv(title, max_results=5)
        except Exception as exc:
            errors.append(f"arXiv lookup failed: {exc}")
            score_info = "; ".join(
                f"{name} best score {score:.2f}" for name, score in scores.items()
            )
            if errors and score_info:
                return None, f"no published match ({score_info}); {'; '.join(errors)}"
            return None, "; ".join(errors)
        best, score = find_best_match_by_title(title, candidates, min_similarity)
        if not best:
            score_info = "; ".join(
                f"{name} best score {score:.2f}" for name, score in scores.items()
            )
            if score_info:
                return None, f"no published match ({score_info}); no arXiv match (best score {score:.2f})"
            return None, f"no arXiv match (best score {score:.2f})"
        candidate_fields = arxiv_entry_to_fields(best)
        candidate_fields = sanitize_fields(candidate_fields)
        candidate_type = entry_type or "article"
        time.sleep(delay)
        return (candidate_type, candidate_fields, True), None

    score_info = "; ".join(
        f"{name} best score {score:.2f}" for name, score in scores.items()
    )
    if errors and score_info:
        return None, f"no match ({score_info}); {'; '.join(errors)}"
    if errors:
        return None, "; ".join(errors)
    return None, f"no match ({score_info})"


def main():
    parser = argparse.ArgumentParser(
        description="Check and fix BibTeX entries by looking up references online."
    )
    parser.add_argument(
        "--input",
        default="project_description/biblio.bib",
        help="Path to the input BibTeX file.",
    )
    parser.add_argument(
        "--output",
        default="project_description/biblio_checked.bib",
        help="Path to the output BibTeX file.",
    )
    parser.add_argument(
        "--mailto",
        default=None,
        help="Email to include in the Crossref User-Agent header.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.88,
        help="Minimum title similarity required to accept a match.",
    )
    parser.add_argument(
        "--crossref-rows",
        type=int,
        default=20,
        help="Number of Crossref results to consider per entry.",
    )
    parser.add_argument(
        "--openalex-rows",
        type=int,
        default=20,
        help="Number of OpenAlex results to consider per entry.",
    )
    parser.add_argument(
        "--semantic-scholar-rows",
        type=int,
        default=20,
        help="Number of Semantic Scholar results to consider per entry.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of entries to process.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        content = handle.read()

    entries = split_entries(content)
    if not entries:
        print(f"No BibTeX entries found in {args.input}", file=sys.stderr)
        sys.exit(1)

    output_entries = []
    updated = []
    unchanged = []
    unresolved = []
    errors = []

    for idx, entry_text in enumerate(entries):
        if args.limit is not None and idx >= args.limit:
            output_entries.append(entry_text)
            continue
        entry_type, key, fields = parse_entry(entry_text)
        display_key = key or f"entry_{idx + 1}"
        print(f"Checking {display_key}", flush=True)
        if not key:
            errors.append((None, "could not parse entry key"))
            output_entries.append(entry_text)
            print("Result: Error: no similar matches found", flush=True)
            continue
        if not entry_type:
            errors.append((key, "could not parse entry type"))
            output_entries.append(entry_text)
            print("Result: Error: no similar matches found", flush=True)
            continue
        candidate, error = process_entry(
            entry_type,
            key,
            fields,
            args.min_similarity,
            args.mailto,
            args.delay,
            args.crossref_rows,
            args.openalex_rows,
            args.semantic_scholar_rows,
        )
        if error:
            unresolved.append((key, error))
            output_entries.append(entry_text)
            if (
                error.startswith("no match")
                or error.startswith("no arXiv match")
                or error.startswith("no Crossref match")
                or error.startswith("no published match")
            ):
                print("Result: Error: no similar matches found", flush=True)
            else:
                print(f"Result: Error: {error}", flush=True)
            continue
        candidate_type, candidate_fields, candidate_is_arxiv = candidate
        merged_fields = merge_fields(fields, candidate_fields)
        if is_arxiv_entry(fields) and not candidate_is_arxiv:
            merged_fields = remove_arxiv_fields(merged_fields)
        merged_fields = sanitize_fields(merged_fields)
        changed_fields = detect_changes(fields, merged_fields)
        if candidate_type != entry_type:
            changed_fields.append("entry_type")
        if changed_fields:
            updated.append((key, sorted(set(changed_fields))))
            print("Result: file modified", flush=True)
        else:
            unchanged.append(key)
            print("Result: file unchanged", flush=True)
        output_entries.append(format_entry(candidate_type, key, merged_fields))

    output_text = "\n\n".join(output_entries).strip() + "\n"
    with open(args.output, "w", encoding="ascii") as handle:
        handle.write(output_text)

    print(f"Total entries: {len(entries)}")
    print(f"Updated: {len(updated)}")
    print(f"Unchanged: {len(unchanged)}")
    print(f"Unresolved: {len(unresolved)}")
    print(f"Errors: {len(errors)}")
    if updated:
        print("\nUpdated entries:")
        for key, fields in updated:
            print(f"- {key}: {', '.join(fields)}")
    if unresolved:
        print("\nUnresolved entries:")
        for key, reason in unresolved:
            print(f"- {key}: {reason}")
    if errors:
        print("\nErrors:")
        for key, reason in errors:
            if key:
                print(f"- {key}: {reason}")
            else:
                print(f"- {reason}")


if __name__ == "__main__":
    main()
