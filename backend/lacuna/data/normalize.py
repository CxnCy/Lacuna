from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_work(work: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields required by Lacuna's temporal network and ML pipeline."""

    topics = []

    for topic in work.get("topics") or []:
        topics.append(
            {
                "id": topic.get("id"),
                "name": topic.get("display_name"),
                "subfield": (topic.get("subfield") or {}).get("display_name"),
                "field": (topic.get("field") or {}).get("display_name"),
                "domain": (topic.get("domain") or {}).get("display_name"),
            }
        )

    author_ids = []

    for authorship in work.get("authorships") or []:
        author_id = (authorship.get("author") or {}).get("id")
        if author_id:
            author_ids.append(author_id)

    return {
        "work_id": work.get("id"),
        "title": work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "topics": topics,
        "author_ids": author_ids,
        "referenced_work_ids": work.get("referenced_works") or [],
        "cited_by_count": work.get("cited_by_count", 0),
    }


def normalize_dataset(
    input_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Load raw OpenAlex works, normalize them, validate them, and save output."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as file:
        raw_works = json.load(file)

    normalized = [normalize_work(work) for work in raw_works]

    valid = [
        work
        for work in normalized
        if work["work_id"]
        and work["publication_year"] is not None
        and work["publication_date"]
        and work["topics"]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(valid, file, ensure_ascii=False)

    return valid