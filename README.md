# MCA Pulse: Lead Intelligence and Market Insight Dashboard

MCA Pulse is a modern Streamlit market-intelligence platform built from the May 2019 Ministry of Corporate Affairs
company registration dataset. Its visual language follows the supplied editorial dashboard reference: an expansive
serif-led hero, compact intelligence cards, numbered narrative sections, warm neutral surfaces, teal signals, and an
optional dark mode.

## Features

- Market intelligence for 11,281 Indian company registrations in May 2019
- City, state and genuine industrial-activity analysis from the source workbook
- Capital hotspots, market clusters, daily formation momentum and geographic ROC view
- Cohort-relative lead scoring with Hot, Warm and Cold opportunities
- RandomForest inference module with lead probability, strength, opportunity score, growth potential and rank
- Modelled website engagement-potential indicators, explicitly separated from observed analytics
- CSV, Excel and priority lead-report downloads scoped to the active filters

## Project Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit market-intelligence interface and exports |
| `utils.py` | Workbook cleaning, feature creation, scoring and Excel export |
| `train_model.py` | RandomForest training pipeline and `model.pkl` persistence |
| `inference_network.py` | Runtime probability and opportunity inference |
| `data/eir_May2019.xlsx` | Supplied MCA workbook copied into the runnable project |
| `model.pkl` | Generated trained inference artifact |
| `reports/training_metrics.json` | Generated holdout metrics and training disclosure |

## Run

Use Python 3.12 or 3.13 on Windows. Python 3.14 is not yet supported by every pinned analytics dependency.
Create a virtual environment, install packages, train the persisted model, and launch Streamlit:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The application also trains `model.pkl` automatically on first launch when the file is absent.

## Data Processing

The app reads the `Indian company registered - May` worksheet, standardizes headers, parses registration dates,
cleans text fields, extracts ROC city names, and converts capital figures from thousand rupees into rupees plus
lakh-friendly display columns. The source also contains `State` and `Industrial Description`, which power real
state and sector insights.

## Scoring And Inference Disclosure

The MCA registration data contains no observed lead conversion, funding outcome or website traffic target. To avoid
inventing such outcomes, `lead_score` is a transparent cohort-relative prioritisation index based on:

- Authorized and paid-up capital strength
- Company class and listed flag
- Within-month registration recency
- ROC-market registration density

The top cohort segment becomes the disclosed proxy target used to train the RandomForest classifier. Consequently,
`lead_probability`, `growth_potential` and engagement-potential values should be used for outreach prioritisation,
not interpreted as measured commercial performance.

In the source period, every Indian-company row is marked `Unlisted`, so listed status has no positive training
variation and should not be treated as a validated predictive factor until additional months or outcomes are added.
