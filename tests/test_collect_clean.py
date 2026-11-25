"""Unit tests for Adike Dhara collection and cleaning modules."""

from __future__ import annotations

import pandas as pd
import pytest

from collect import COLUMNS, _canonical_frame, from_csv
from clean import clean, flag_missing_days, parse_dates, remove_outliers


def test_canonical_columns_constant():
    assert COLUMNS == [
        "date",
        "market",
        "variety",
        "min_price",
        "max_price",
        "modal_price",
        "arrivals",
    ]


def test_canonical_frame_handles_aliases():
    rows = [
        {
            "date": "01-01-2025",
            "market": "APMC Yard, Davanagere",
            "variety": "Moati",
            "min_price": 400.0,
            "max_price": 450.0,
            "modal_price": 420.0,
            "arrivals": 300,
        }
    ]
    df = _canonical_frame(rows)
    assert list(df.columns) == COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["variety"] == "Moati"
    assert df.iloc[0]["modal_price"] == 420.0


def test_from_csv_missing_required(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    res = from_csv(bad)
    assert res.ok is False
    assert res.data.empty


def test_parse_dates_dayfirst():
    df = pd.DataFrame({"date": ["01-02-2025", "15-03-2025"], "x": [1, 2]})
    out = parse_dates(df)
    assert out["date"].iloc[0].month == 2  # day-first: Feb 1
    assert out["date"].iloc[1].month == 3


def test_remove_outliers_flags_extreme():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=10),
            "market": ["M"] * 10,
            "variety": ["V"] * 10,
            "modal_price": [100, 102, 98, 101, 99, 103, 100, 97, 101, 1000],
        }
    )
    out = remove_outliers(df)
    flagged = out[out["outlier"] == True]
    assert len(flagged) == 1
    assert pd.isna(flagged["modal_price"].iloc[0])


def test_flag_missing_days_inserts_gaps():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-03"]),
            "market": ["M", "M"],
            "variety": ["V", "V"],
            "modal_price": [100.0, 102.0],
        }
    )
    out = flag_missing_days(df)
    assert (out["date"] == pd.Timestamp("2025-01-02")).any()
    missing = out.loc[out["date"] == pd.Timestamp("2025-01-02"), "missing_day"].iloc[0]
    assert bool(missing) is True


def test_clean_writes_processed_csv(tmp_path):
    raw = tmp_path / "raw.csv"
    rows = []
    for i in range(5):
        rows.append(
            {
                "date": f"0{i+1}-01-2025",
                "market": "M",
                "variety": "V",
                "min_price": 100.0,
                "max_price": 110.0,
                "modal_price": 105.0,
                "arrivals": 100,
            }
        )
    pd.DataFrame(rows).to_csv(raw, index=False)
    out_dir = tmp_path / "processed"
    result = clean(raw, output_dir=str(out_dir), variety="V", write_csv=True)
    assert not result.empty
    written = list(out_dir.glob("*.csv"))
    assert len(written) == 1
