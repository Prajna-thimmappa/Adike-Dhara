# Adike Dhara (ಅಡಿಕೆ ಧಾರಣೆ)

**Areca Price Tracker & Forecast**

> Know today's rate. Plan tomorrow's sale.

Adike Dhara collects daily arecanut (betel nut /.supari) prices from Karnataka
mandis, cleans and stores them, trains short-term time-series forecasting
models, and presents everything in a Streamlit dashboard.

## Project structure

```
adike-dhara/
├── .streamlit/
│   └── config.toml        # green/brown theme
├── data/
│   ├── raw/               # raw downloads (not committed)
│   └── processed/         # cleaned daily price CSVs (not committed)
├── notebooks/
│   └── eda.ipynb          # exploration: trends, seasonality, missing days
├── src/
│   ├── collect.py         # fetch prices; CSV + optional web scraper
│   ├── clean.py           # parse dates, flag missing days, remove outliers
│   ├── features.py        # lag / rolling / calendar features, no leakage
│   ├── models.py          # naive, MA7, SARIMA, Prophet, XGBoost forecasters
│   ├── evaluate.py        # time-based split, walk-forward, MAE/RMSE/MAPE
│   └── forecast.py        # 7 & 14 day forecasts with intervals
├── app.py                 # Streamlit dashboard
├── .github/workflows/update.yml  # scheduled daily update
├── requirements.txt
├── README.md
└── .gitignore
```

## Tech stack

- Python 3.10+
- pandas, numpy, requests, beautifulsoup4
- statsmodels (SARIMA), prophet, xgboost, scikit-learn
- Streamlit + Plotly

Versions are pinned in `requirements.txt`.

## Where to get data

Adike Dhara works with any export that has a date plus a price column. Recommended
public sources for Karnataka arecanut prices:

- **Agmarknet** (agmarknet.gov.in) – daily APMC modal prices by market/commodity
- **data.gov.in** – government open-data price datasets
- **CAMPCO** (campco.co.in) – arecanut growers' cooperative price bulletins

Export to CSV with columns like `date, market, variety, min_price, max_price,
modal_price, arrivals` and place it in `data/raw/`. A manual-entry CSV is the
recommended fallback when no public scrape target is available.

> The optional web scraper in `src/collect.py` respects `robots.txt`, uses a
> clear `User-Agent`, and paces requests. Only scrape pages whose terms of
> service permit it.

## Setup

```bash
# Clone
git clone https://github.com/Prajna-thimmappa/Adike-Dhara.git
cd Adike-Dhara

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline

### 1. Generate / place raw data

A sample dataset is included for testing:

```bash
python - <<'PY'
# or use your own Agmarknet / data.gov.in / CAMPCO export
PY
python -m src.collect --source csv --path data/raw/sample_arecanut.csv
```

### 2. Clean the data

Produces one tidy CSV per variety in `data/processed/`:

```bash
python -m src.clean data/raw/sample_arecanut.csv
```

Missing days (Sundays/holidays) are flagged rather than invented; outliers are
flagged via an IQR fence.

### 3. Build features

```bash
python -c "
from src.features import load_processed, build_features
df = load_processed('data/processed/<variety>_daily.csv')
X, y = build_features(df)
"
```

### 4. Train & evaluate

Walk-forward validation with MAE / RMSE / MAPE; saves `data/processed/evaluation.csv`.

```bash
python -m src.evaluate
```

A model counts as useful only if it beats both baselines (naive and 7-day MA).

### 5. Forecast

7- and 14-day forecasts with prediction intervals from the best model:

```bash
python -m src.forecast data/processed/<variety>_daily.csv \
  data/raw/sample_arecanut.csv <variety> "<market>"
```

### 6. Run the dashboard

```bash
streamlit run app.py
```

## Scheduled updates

`.github/workflows/update.yml` runs collection + cleaning daily at 02:00 UTC and
commits updated processed data. Enable Actions in the repository settings.

## Testing

Run the unit test suite with `pytest`:

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests cover collection aliasing, date parsing, outlier flagging, missing-day
insertion, feature construction without future leakage, chronological splits,
walk-forward splits, and the baseline forecasters.

## Docker

Build and run the container (includes a sample dataset and a pre-cleaned
processed file):

```bash
docker build -t adike-dhara .
docker run -p 8501:8501 adike-dhara
```

Then open http://localhost:8501.

## Limitations

- **Unpredictable shocks:** Arecanut prices can move suddenly on government
  policy, import decisions, weather, festivals, or CAMPCO interventions that no
  history-based model can anticipate.
- **Limited data:** Public daily price series for many Karnataka mandis are
  short or incomplete. Forecast accuracy improves with longer, cleaner history.
- **Missing days:** Mandis close on Sundays and holidays; these are flagged,
  not interpolated into the raw record, so model inputs use interpolation only
  internally.
- **Not financial advice:** Forecasts are estimates. Verify against the mandi
  rate before making sale decisions.

## License

MIT
