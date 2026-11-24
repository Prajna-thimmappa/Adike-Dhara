"""Data cleaning for Adike Dhara.

Turns raw collected prices into one tidy CSV per variety:

* Parse dates robustly.
* Flag missing days (holidays / Sundays) instead of inventing prices.
* Remove obvious outliers (prices beyond a configurable IQR fence).
* Sort and deduplicate.
* Write ``data/processed/<variety>_daily.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

CANONICAL_COLUMNS: list[str] = [
    "date",
    "market",
    "variety",
    "min_price",
    "max_price",
    "modal_price",
    "arrivals",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a collected raw CSV into a DataFrame with canonical columns."""
    df = pd.read_csv(path)
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[CANONICAL_COLUMNS].copy()


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the date column and drop rows with unparseable dates.

    Tries day-first (Agmarknet / data.gov.in style dd-mm-yyyy) first, then
    falls back to a general parse, so both exported formats are handled.
    """
    df = df.copy()
    parsed = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    na_mask = parsed.isna()
    if na_mask.any():
        fallback = pd.to_datetime(df.loc[na_mask, "date"], errors="coerce")
        parsed.loc[na_mask] = fallback
    before = len(df)
    df = df.assign(date=parsed).dropna(subset=["date"])
    dropped = before - len(df)
    if dropped:
        print(f"[clean] Dropped {dropped} rows with unparseable dates")
    return df


def remove_outliers(
    df: pd.DataFrame,
    *,
    column: str = "modal_price",
    k: float = 3.0,
    group_by: tuple[str, ...] = ("market", "variety"),
) -> pd.DataFrame:
    """Remove obvious outliers using a per-group IQR fence.

    Rows beyond ``Q1 - k*IQR`` or ``Q3 + k*IQR`` are flagged (not silently
    deleted) by setting the price to NaN and recording an ``outlier`` flag.
    Uses ``transform`` so no columns are lost.
    """
    df = df.copy()
    df["outlier"] = False

    g = df.groupby(list(group_by))[column]
    q1 = g.transform("quantile", 0.25)
    q3 = g.transform("quantile", 0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    # Only flag when the group has real spread (IQR > 0).
    mask = mask & (iqr > 0)

    df.loc[mask, "outlier"] = True
    df.loc[mask, column] = pd.NA
    return df.reset_index(drop=True)


def flag_missing_days(
    df: pd.DataFrame,
    *,
    group_by: tuple[str, ...] = ("market", "variety"),
) -> pd.DataFrame:
    """Reindex each group to a full daily calendar and flag missing days.

    Missing days (holidays, Sundays) are kept as NaN prices with
    ``missing_day = True`` so downstream steps never invent data. The original
    observed days keep ``missing_day = False``.
    """
    df = df.copy()
    df["missing_day"] = False

    frames: list[pd.DataFrame] = []
    for _, group in df.groupby(list(group_by), group_keys=False):
        group_vals = {col: group[col].iloc[0] for col in group_by}
        group = group.sort_values("date")
        full_index = pd.date_range(start=group["date"].min(), end=group["date"].max(), freq="D")
        group = group.set_index("date").reindex(full_index)
        group.index.name = "date"
        group = group.reset_index()
        # Fill grouping columns for calendar rows that were inserted.
        for col, val in group_vals.items():
            group[col] = group[col].fillna(val)
        # Mark rows that were not originally observed.
        observed = set(df.set_index(["date", *group_by]).index)
        group["missing_day"] = ~group.set_index(["date", *group_by]).index.isin(observed)
        frames.append(group)

    if not frames:
        return df
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(list(group_by) + ["date"]).reset_index(drop=True)
    return result


def clean(
    raw_path: str | Path,
    *,
    output_dir: str | Path = "data/processed",
    variety: str | None = None,
    k: float = 3.0,
    write_csv: bool = True,
) -> pd.DataFrame:
    """Run the full cleaning pipeline and (optionally) write one CSV per variety.

    Parameters
    ----------
    raw_path:
        Path to the raw collected CSV.
    output_dir:
        Directory for processed CSVs.
    variety:
        If set, only clean this variety (exact match on the ``variety`` column).
    k:
        IQR multiplier for outlier detection.
    write_csv:
        Whether to write ``<variety>_daily.csv`` per variety.

    Returns
    -------
    pd.DataFrame
        The cleaned frame with ``missing_day`` and ``outlier`` flag columns.
    """
    df = load_raw(raw_path)
    df = parse_dates(df)

    if variety is not None:
        df = df[df["variety"] == variety].copy()
        if df.empty:
            raise ValueError(f"No rows found for variety: {variety!r}")

    df = remove_outliers(df, k=k)
    df = flag_missing_days(df)

    # Final tidy column order.
    keep = CANONICAL_COLUMNS + ["missing_day", "outlier"]
    df = df[[c for c in keep if c in df.columns]]

    if write_csv:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if variety is not None:
            varieties = [variety]
        else:
            varieties = df["variety"].dropna().unique()
        for var in varieties:
            var_df = df[df["variety"] == var]
            safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(var))
            out_path = out_dir / f"{safe_name}_daily.csv"
            var_df.to_csv(out_path, index=False)
            print(f"[clean] Wrote {len(var_df)} rows -> {out_path}")

    return df


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_arecanut.csv"
    clean(target)
