FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data is not committed; create a sample for a smoke test.
RUN python - <<'PY'
from pathlib import Path
import pandas as pd, numpy as np
from datetime import date
rng = np.random.default_rng(42)
markets = ["APMC Yard, Davanagere", "APMC Yard, Shimoga", "APMC Yard, Chikkamagaluru"]
varieties = ["Moati", "Bhasuri", "Sahi"]
start, end = date(2025, 1, 1), date(2025, 3, 31)
days = pd.date_range(start, end, freq="D")
base = {"Moati": 420, "Bhasuri": 450, "Sahi": 400}
rows = []
for m in markets:
    for v in varieties:
        level = base[v]
        for d in days:
            if d.weekday() == 6 or rng.random() < 0.05:
                continue
            level += rng.normal(0, 6)
            modal = max(200, level)
            rows.append({
                "date": d.strftime("%d-%m-%Y"),
                "market": m, "variety": v,
                "min_price": round(modal * 0.95, 2),
                "max_price": round(modal * 1.05, 2),
                "modal_price": round(modal, 2),
                "arrivals": int(rng.integers(50, 500)),
            })
Path("data/raw").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("data/raw/sample_arecanut.csv", index=False)
PY

RUN python -m src.clean data/raw/sample_arecanut.csv

EXPOSE 8501
HEALTHCHECK CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/')"] || true

ENTRYPOINT ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"]
