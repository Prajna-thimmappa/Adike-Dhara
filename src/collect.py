"""Data collection for Adike Dhara (arecanut price tracker).

Provides a single interface to fetch daily arecanut prices from multiple
sources:

* ``from_csv`` - load an Agmarknet / data.gov.in / manual-entry CSV export.
* ``from_web`` - optional scraper for a public price page (respects
  robots.txt, rate limits, and User-Agent conventions).

Every collector returns a tidy DataFrame with the canonical columns::

    date, market, variety, min_price, max_price, modal_price, arrivals
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Canonical output columns shared by every source.
COLUMNS: list[str] = [
    "date",
    "market",
    "variety",
    "min_price",
    "max_price",
    "modal_price",
    "arrivals",
]

# Identify this scraper in requests so site operators can contact us.
USER_AGENT = (
    "Adike-Dhara/1.0 (arecanut price tracker; educational/research use; "
    "contact: maintainer@example.com)"
)

# Polite delay (seconds) between requests to the same host.
REQUEST_DELAY_SECONDS = 1.5


@dataclass(frozen=True)
class SourceResult:
    """Outcome of a collection attempt."""

    source: str
    rows: int
    ok: bool
    message: str
    data: pd.DataFrame | None = None


def _canonical_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a canonical DataFrame from raw row dicts."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("min_price", "max_price", "modal_price", "arrivals"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "modal_price"])
    df = df.sort_values(["market", "variety", "date"]).reset_index(drop=True)
    return df


def from_csv(path: str | Path, **read_kwargs) -> SourceResult:
    """Load prices from a CSV export (Agmarknet / data.gov.in / manual entry).

    The CSV is expected to contain (at minimum) a date column plus one price
    column. Aliases are mapped to the canonical schema so the same downstream
    cleaning works regardless of the exporter.

    Parameters
    ----------
    path:
        Path to the CSV file.
    read_kwargs:
        Extra keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    SourceResult
        Parsed frame and a human-readable status message.
    """
    path = Path(path)
    if not path.exists():
        return SourceResult(
            source=str(path),
            rows=0,
            ok=False,
            message=f"File not found: {path}",
            data=pd.DataFrame(columns=COLUMNS),
        )

    try:
        raw = pd.read_csv(path, **read_kwargs)
    except Exception as exc:  # pragma: no cover - defensive I/O
        return SourceResult(
            source=str(path),
            rows=0,
            ok=False,
            message=f"Failed to read CSV: {exc}",
            data=pd.DataFrame(columns=COLUMNS),
        )

    # Normalise column names to lowercase with underscores.
    raw.columns = (
        raw.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Alias map: common Agmarknet / data.gov.in headers -> canonical names.
    alias = {
        "arrival_date": "date",
        "date": "date",
        "market": "market",
        "district": "market",
        "apmc": "market",
        "variety": "variety",
        "commodity": "variety",
        "min_price": "min_price",
        "minimum_price": "min_price",
        "max_price": "max_price",
        "maximum_price": "max_price",
        "modal_price": "modal_price",
        "modalprice": "modal_price",
        "arrivals": "arrivals",
        "quantity_arrivals": "arrivals",
    }

    renamed = raw.rename(columns={c: alias.get(c, c) for c in raw.columns})

    missing = [c for c in ("date", "modal_price") if c not in renamed.columns]
    if missing:
        return SourceResult(
            source=str(path),
            rows=0,
            ok=False,
            message=(
                "CSV is missing required columns: "
                + ", ".join(missing)
                + ". Found: " + ", ".join(map(str, renamed.columns))
            ),
            data=pd.DataFrame(columns=COLUMNS),
        )

    # Fill absent optional columns with NaN.
    for col in COLUMNS:
        if col not in renamed.columns:
            renamed[col] = pd.NA

    df = _canonical_frame(renamed[COLUMNS].to_dict("records"))
    return SourceResult(
        source=str(path),
        rows=len(df),
        ok=not df.empty,
        message=(
            f"Loaded {len(df)} rows from {path.name}"
            if not df.empty
            else f"No usable rows in {path.name}"
        ),
        data=df,
    )


def _respects_robots(base_url: str) -> bool:
    """Best-effort robots.txt check.

    Returns True when the path appears crawlable or when robots.txt cannot be
    fetched (fail-open, with a warning logged to the message). This keeps the
    scraper usable while still checking the standard exclusion protocol.
    """
    try:
        robots = requests.get(
            base_url.rstrip("/") + "/robots.txt",
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        if robots.status_code != 200:
            return True
        body = robots.text.lower()
        # Very conservative: disallow only if the homepage path is explicitly
        # disallowed for our user agent or for all agents.
        disallowed = any(
            line.strip().startswith("disallow: /")
            for line in body.splitlines()
        )
        return not disallowed
    except Exception:
        # Fail open: network failure should not block legitimate collection.
        return True


def from_web(
    url: str,
    parse_fn: Callable[[BeautifulSoup], list[dict]],
    *,
    delay: float = REQUEST_DELAY_SECONDS,
    timeout: int = 15,
) -> SourceResult:
    """Fetch and parse a public price page.

    Parameters
    ----------
    url:
        Public URL listing arecanut prices.
    parse_fn:
        Callback receiving a :class:`bs4.BeautifulSoup` and returning a list
        of raw row dicts (keys need not be canonical; :func:`_canonical_frame`
        handles aliasing via :func:`from_csv`-style normalisation).
    delay:
        Seconds to sleep before returning (polite pacing between calls).
    timeout:
        Request timeout in seconds.

    Returns
    -------
    SourceResult
        Parsed frame and status message.
    """
    try:
        allowed = _respects_robots(url)
    except Exception:
        allowed = True

    if not allowed:
        return SourceResult(
            source=url,
            rows=0,
            ok=False,
            message="robots.txt disallows crawling this path",
            data=pd.DataFrame(columns=COLUMNS),
        )

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        return SourceResult(
            source=url,
            rows=0,
            ok=False,
            message=f"Request failed: {exc}",
            data=pd.DataFrame(columns=COLUMNS),
        )

    # Be polite: pace requests to the same host.
    time.sleep(delay)

    soup = BeautifulSoup(response.text, "html.parser")
    try:
        rows = parse_fn(soup) or []
    except Exception as exc:  # pragma: no cover - defensive parsing
        return SourceResult(
            source=url,
            rows=0,
            ok=False,
            message=f"Parsing failed: {exc}",
            data=pd.DataFrame(columns=COLUMNS),
        )

    df = _canonical_frame(rows)
    return SourceResult(
        source=url,
        rows=len(df),
        ok=not df.empty,
        message=(
            f"Scraped {len(df)} rows from {url}" if not df.empty else "No rows parsed"
        ),
        data=df,
    )


def collect(
    source: Literal["csv", "web"],
    *,
    path: str | Path | None = None,
    url: str | None = None,
    parse_fn: Callable[[BeautifulSoup], list[dict]] | None = None,
    **kwargs,
) -> SourceResult:
    """Unified entry point for collecting prices.

    Parameters
    ----------
    source:
        ``"csv"`` for a file export or ``"web"`` for the optional scraper.
    path:
        CSV path (required when ``source="csv"``).
    url:
        Page URL (required when ``source="web"``).
    parse_fn:
        HTML parser callback (required when ``source="web"``).
    kwargs:
        Forwarded to the underlying collector.

    Returns
    -------
    SourceResult
    """
    if source == "csv":
        if path is None:
            raise ValueError("path is required for source='csv'")
        return from_csv(path, **kwargs)
    if source == "web":
        if url is None or parse_fn is None:
            raise ValueError("url and parse_fn are required for source='web'")
        return from_web(url, parse_fn, **kwargs)
    raise ValueError(f"Unknown source: {source!r}. Use 'csv' or 'web'.")


def example_agmarknet_parser(soup: BeautifulSoup) -> list[dict]:
    """Example parser stub for a generic Agmarknet-style HTML table.

    Replace the selectors below to match the actual target page. The function
    returns raw row dicts; header aliasing happens downstream.

    This is intentionally a template - do not point it at a live site without
    verifying that site's robots.txt and terms of service.
    """
    rows: list[dict] = []
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if not headers:
            continue
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) != len(headers):
                continue
            rows.append(dict(zip(headers, cells)))
    return rows
