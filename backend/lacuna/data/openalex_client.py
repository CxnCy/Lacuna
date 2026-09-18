from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx


OPENALEX_BASE_URL = "https://api.openalex.org"


class OpenAlexClient:
    """Client for downloading a bounded OpenAlex development corpus."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make an OpenAlex request with simple retry handling."""

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(
                        f"{OPENALEX_BASE_URL}/works",
                        params=params,
                    )
                    response.raise_for_status()
                    return response.json()

            except (httpx.HTTPError, httpx.TimeoutException):
                if attempt == self.max_retries:
                    raise

                time.sleep(attempt * 2)

        raise RuntimeError("OpenAlex request failed unexpectedly.")

    def fetch_works(
        self,
        start_year: int = 2005,
        end_year: int = 2025,
        max_works: int = 1000,
        per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch a bounded corpus using cursor pagination."""

        works: list[dict[str, Any]] = []
        cursor = "*"

        while len(works) < max_works:
            params = {
                "filter": (
                    f"from_publication_date:{start_year}-01-01,"
                    f"to_publication_date:{end_year}-12-31"
                ),
                "per-page": min(per_page, max_works - len(works)),
                "cursor": cursor,
            }

            data = self._request(params)
            results = data.get("results", [])

            if not results:
                break

            works.extend(results)

            cursor = data.get("meta", {}).get("next_cursor")

            if not cursor:
                break

        return works[:max_works]

    def fetch_and_cache(
        self,
        output_path: str | Path,
        start_year: int = 2005,
        end_year: int = 2025,
        max_works: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch works and save the raw API response locally."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        works = self.fetch_works(
            start_year=start_year,
            end_year=end_year,
            max_works=max_works,
        )

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(works, file, ensure_ascii=False)

        return works